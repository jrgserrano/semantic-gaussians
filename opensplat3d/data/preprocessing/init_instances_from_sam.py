"""
init_instances_from_sam.py
───────────────────────────────────────────────────────────────
Assigns SAM instance IDs to each 3-D point in points3d.ply by
projecting the point cloud onto every available training view,
reading the SAM mask at the projected pixel, and assigning the
majority-voted instance ID.

The result is saved as:
  <dataset_path>/points3d_instances.npz
    ├─ xyz          (N, 3)  float32  – 3-D coordinates
    ├─ rgb          (N, 3)  uint8    – original colours
    ├─ instance_id  (N,)    int32    – majority-voted SAM ID
    └─ vote_count   (N,)    int32    – how many views agreed

Usage:
  uv run python opensplat3d/data/preprocessing/init_instances_from_sam.py \\
      /home/ubuntu/datasets/Replica/office0 \\
      --nth-frames 5 \\
      --mask-level default
"""

import argparse
import struct
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm


# ─── PLY loader (reads only x, y, z, red, green, blue) ────────────────────────

def load_ply(ply_path: Path):
    """Read vertex x,y,z + r,g,b from a binary little-endian PLY file."""
    with open(ply_path, "rb") as f:
        # Parse header
        header_lines = []
        while True:
            line = f.readline().decode("utf-8", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        num_vertices = 0
        prop_names = []
        for line in header_lines:
            if line.startswith("element vertex"):
                num_vertices = int(line.split()[-1])
            if line.startswith("property"):
                prop_names.append(line.split()[-1])

        # Build struct format – assume all properties are float32 except uchar ones
        fmt_map = {"float": "f", "uchar": "B", "int": "i", "double": "d",
                   "int16": "h", "uint16": "H", "int32": "i", "uint32": "I"}
        type_map = {}
        for line in header_lines:
            if line.startswith("property"):
                parts = line.split()
                type_map[parts[-1]] = parts[-2]

        struct_fmt = "<" + "".join(fmt_map.get(type_map.get(p, "float"), "f") for p in prop_names)
        vertex_size = struct.calcsize(struct_fmt)

        xyz = np.zeros((num_vertices, 3), dtype=np.float32)
        rgb = np.zeros((num_vertices, 3), dtype=np.uint8)

        xi = prop_names.index("x")
        yi = prop_names.index("y")
        zi = prop_names.index("z")
        ri = prop_names.index("red") if "red" in prop_names else None
        gi = prop_names.index("green") if "green" in prop_names else None
        bi = prop_names.index("blue") if "blue" in prop_names else None

        data = f.read(vertex_size * num_vertices)
        unpacked = struct.iter_unpack(struct_fmt, data)
        for i, row in enumerate(unpacked):
            xyz[i] = (row[xi], row[yi], row[zi])
            if ri is not None:
                rgb[i] = (row[ri], row[gi], row[bi])

    return xyz, rgb


# ─── Camera helpers ────────────────────────────────────────────────────────────

def parse_traj_txt(path: Path):
    c2ws = []
    with open(path) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 16:
                c2ws.append(np.array(vals, dtype=np.float64).reshape(4, 4))
    return c2ws


def project_points(xyz: np.ndarray, c2w: np.ndarray, fx: float, fy: float,
                   cx: float, cy: float, width: int, height: int):
    """
    Projects an (N, 3) world-space array onto a camera defined by its
    camera-to-world matrix c2w.  Returns (u, v, visible_mask).
    """
    w2c = np.linalg.inv(c2w)

    # Transform to camera space
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float64)])  # (N, 4)
    xyz_cam = (w2c @ xyz_h.T).T  # (N, 4)

    z = xyz_cam[:, 2]
    in_front = z > 0.01  # ignore points behind or at the camera

    u = np.zeros(len(xyz), dtype=np.float32)
    v = np.zeros(len(xyz), dtype=np.float32)

    u[in_front] = fx * xyz_cam[in_front, 0] / z[in_front] + cx
    v[in_front] = fy * xyz_cam[in_front, 1] / z[in_front] + cy

    # Visibility: in front, and within image bounds
    visible = (
        in_front
        & (u >= 0) & (u < width - 1)
        & (v >= 0) & (v < height - 1)
    )
    return u.astype(np.int32), v.astype(np.int32), visible


# ─── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    dataset_path = Path(args.dataset_path)
    ply_path = dataset_path / "points3d.ply"
    traj_path = dataset_path / "traj.txt"
    sam_dir = dataset_path / args.sam_subdir
    output_path = dataset_path / "points3d_instances.npz"

    if not ply_path.exists():
        raise FileNotFoundError(f"points3d.ply not found at {ply_path}")
    if not traj_path.exists():
        raise FileNotFoundError(f"traj.txt not found at {traj_path}")

    print(f"Loading point cloud from {ply_path} ...")
    xyz, rgb = load_ply(ply_path)
    N = len(xyz)
    print(f"  {N:,} points loaded.")

    print(f"Loading camera trajectories from {traj_path} ...")
    c2ws = parse_traj_txt(traj_path)
    total_frames = len(c2ws)
    frame_indices = list(range(0, total_frames, args.nth_frames))
    print(f"  {total_frames} total frames -> using every {args.nth_frames}th = {len(frame_indices)} frames.")

    # Camera intrinsics (same as replica_reader.py / extract_replica_pcd.py)
    width, height = 1200, 680
    fx = fy = 600.0
    cx, cy = width / 2.0, height / 2.0

    # Vote accumulation: for each point a Counter of SAM instance IDs
    # We use a compact (N, max_instances) approach instead of N Counters for speed
    # => we accumulate a (N,) best_id and (N,) best_count via flat arrays
    vote_counts = np.zeros((N,), dtype=np.int32)       # max votes so far
    best_ids    = np.full((N,), -1, dtype=np.int32)    # best SAM ID so far
    tmp_counts  = {}                                    # point_idx -> Counter (fallback)

    missing_masks = 0

    for frame_idx in tqdm(frame_indices, desc="Projecting & Voting"):
        mask_path = sam_dir / f"frame{frame_idx:06d}.npz"
        if not mask_path.exists():
            missing_masks += 1
            continue

        with np.load(mask_path) as data:
            sam_mask = data[args.mask_level].astype(np.int32)  # (H, W)

        c2w = c2ws[frame_idx]
        u, v, vis = project_points(xyz, c2w, fx, fy, cx, cy, width, height)

        # For all visible points, look up their mask ID
        vis_idx = np.where(vis)[0]
        if len(vis_idx) == 0:
            continue

        sampled_ids = sam_mask[v[vis_idx], u[vis_idx]]  # (M,)

        # Fast majority vote update:
        # Instead of a slow python loop, group by point index and update best_id
        # when the new id equals existing best (increment count) or is new (needs Counter).
        # We do a vectorised pass: for each point check if sampled_id == best_ids
        matches_best  = sampled_ids == best_ids[vis_idx]        # (M,) bool
        new_winner    = ~matches_best & (sampled_ids >= 0)      # id is valid (>=0 = labelled)

        # Increment counter for those that match existing best
        vote_counts[vis_idx[matches_best]] += 1

        # For new/different ids we need per-point Counters
        for pt, sid in zip(vis_idx[new_winner], sampled_ids[new_winner]):
            if pt not in tmp_counts:
                # Seed the counter with the current best
                tmp_counts[pt] = Counter({int(best_ids[pt]): int(vote_counts[pt])})
            tmp_counts[pt][int(sid)] += 1
            # Update best if this id overtook the current leader
            curr_best_id, curr_best_cnt = tmp_counts[pt].most_common(1)[0]
            best_ids[pt]    = curr_best_id
            vote_counts[pt] = curr_best_cnt

    if missing_masks:
        print(f"  Warning: {missing_masks} frames had no SAM mask and were skipped.")

    unlabelled = (best_ids == -1).sum()
    print(f"\nVoting done.")
    print(f"  Labelled points : {N - unlabelled:,} / {N:,}")
    print(f"  Unlabelled (id=-1): {unlabelled:,}")

    np.savez_compressed(
        output_path,
        xyz=xyz.astype(np.float32),
        rgb=rgb,
        instance_id=best_ids,
        vote_count=vote_counts,
    )
    print(f"\nSaved instance-labelled cloud to: {output_path}")

    # Print instance summary
    unique_ids, counts = np.unique(best_ids[best_ids >= 0], return_counts=True)
    print(f"  Unique SAM instances found: {len(unique_ids)}")
    print(f"  Top 10 by point count:")
    top10 = np.argsort(counts)[::-1][:10]
    for rank, i in enumerate(top10):
        print(f"    #{rank+1}  instance_id={unique_ids[i]}  points={counts[i]:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Label 3-D points with SAM instance IDs via majority voting."
    )
    parser.add_argument("dataset_path", type=str,
                        help="Path to Replica scene dir (contains points3d.ply, traj.txt, sam/)")
    parser.add_argument("--nth-frames", type=int, default=5,
                        help="Use every Nth frame (default: 5 -> 400 frames for a 2000-frame scene)")
    parser.add_argument("--mask-level", type=str, default="default",
                        help="SAM mask level to use (default: 'default')")
    parser.add_argument("--sam-subdir", type=str, default="sam",
                        help="Subdirectory containing SAM .npz files (default: 'sam')")
    args = parser.parse_args()
    main(args)
