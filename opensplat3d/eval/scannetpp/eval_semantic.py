# Call semantic evaluation but do it also per scene and store the results in a file
import json
import os
import time
from pathlib import Path

import numpy as np
from scannetpp.common.file_io import load_yaml_munch, read_txt_list
from scannetpp.semantic.eval.eval_semantic import eval_semantic
from scannetpp.semantic.utils.confmat import ConfMat


def write_conf_mat(confmat: ConfMat, semantic_classes: list[str]) -> list[str]:
    lines = []
    sep = " | "

    line = ""
    line += "{:<20}".format("class") + sep
    for class_name in semantic_classes:
        line += "{:<20}".format(class_name) + sep
    lines.append(line)
    lines.append("#" * len(line))

    for i, class_name in enumerate(semantic_classes):
        line = "{:<20}".format(class_name) + sep
        for j in range(len(semantic_classes)):
            line += "{:>20}".format(confmat.mat[i, j].astype(np.int32)) + sep
        lines.append(line)

    return lines


def write_results(
    confmats: dict[int, ConfMat], semantic_classes: list[str]
) -> list[str]:
    lines = []

    for k, confmat in confmats.items():
        lines.append(
            f"Top {k} mIOU: {confmat.miou:>10.5f} | mAcc: {np.nanmean(confmat.accs / 100):>10.5f}"
        )
        for class_name, class_iou, class_acc in zip(
            semantic_classes, confmat.ious, confmat.accs
        ):
            lines.append(
                f"{class_name: <25}: {class_iou:>10.5f} | {class_acc / 100:>10.5f}"
            )
        lines.append("----------------------------------------------------")
        lines.append("")

        lines += write_conf_mat(confmat, semantic_classes)
        lines.append("")

    return lines


def write_json_results(confmats: dict[int, ConfMat], semantic_classes: list[str]):
    results = {}
    for k, confmat in confmats.items():
        results[k] = {
            "k": k,
            "mIOU": confmat.miou,
            "mAcc": np.nanmean(confmat.accs / 100),
            "classes": {},
            # "ious": confmat.ious.tolist(),
            # "accs": (confmat.accs / 100).tolist(),
            # "confmat": confmat.mat.tolist(),
            "semantic_classes": semantic_classes,
        }

        for class_name, class_iou, class_acc in zip(
            semantic_classes, confmat.ious, confmat.accs
        ):
            results[k]["classes"][class_name] = {
                "iou": class_iou,
                "acc": class_acc / 100,
            }

        results[k]["confmat"] = confmat.mat.tolist()

    return results


def main(preds_dir: Path, config_file: Path, scene_id: str | None = None):
    cfg = load_yaml_munch(config_file)
    cfg.scene_list_file = os.path.expandvars(cfg.scene_list_file)
    cfg.gt_dir = os.path.expandvars(cfg.gt_dir)
    cfg.data_root = os.path.expandvars(cfg.data_root)
    cfg.classes_file = os.path.expandvars(cfg.classes_file)

    is_eval_run = (preds_dir.parent.parent / "scenes").exists()
    if not is_eval_run and scene_id is None:
        # {exp_dir}/eval_predictions/semantic/{pred_dir}
        pred_files = list((preds_dir / "semantic").glob("*.txt"))
        assert len(pred_files) == 1, (
            f"Expected exactly one prediction file in {preds_dir / 'semantic'}, found {len(pred_files)}"
        )
        scene_id = pred_files[0].stem
    assert is_eval_run or scene_id is not None, (
        "Expected eval run structure with 'scenes' directory two levels up or a single scene_id."
    )
    output_dir = preds_dir.parent.parent / "eval_results" / preds_dir.name

    preds_dir = preds_dir / "semantic"
    output_dir = output_dir / "semantic"
    cfg.preds_dir = str(preds_dir)

    print(f"Using predictions from: {preds_dir}")
    print(f"Output directory: {output_dir}")

    all_scene_ids = read_txt_list(cfg.scene_list_file)
    if scene_id is not None:
        assert scene_id in all_scene_ids, (
            f"The provided scene id '{scene_id}' is not in the list."
        )
        all_scene_ids = [scene_id]
    semantic_classes = read_txt_list(cfg.classes_file)
    num_classes = len(semantic_classes)

    eval_against_gt = cfg.preds_dir == cfg.gt_dir
    if eval_against_gt:
        print("Evaluating against GT")

    scene_output_dir = output_dir / "scenes"
    scene_output_dir.mkdir(parents=True, exist_ok=True)

    for i, scene_ids in enumerate([[x] for x in all_scene_ids] + [all_scene_ids]):
        start = time.time()
        is_all = i == len(all_scene_ids)
        scene_id = scene_ids[0] if not is_all else "all"

        confmats = eval_semantic(
            scene_ids,
            cfg.preds_dir,
            cfg.gt_dir,
            cfg.data_root,
            num_classes,
            -1,
            [1, 3],
            eval_against_gt=eval_against_gt,
        )

        lines = write_results(confmats, semantic_classes)
        if is_all:
            for k, confmat in confmats.items():
                print(
                    f"Top {k} mIOU: {confmat.miou:>10.5f} | mAcc: {np.nanmean(confmat.accs / 100):>10.5f}"
                )

                for class_name, class_iou, class_acc in zip(
                    semantic_classes, confmat.ious, confmat.accs
                ):
                    print(
                        f"{class_name: <25}: {class_iou:>10.5f} | {class_acc / 100:>10.5f}"
                    )

                print("----------------------------------------------------")
            scene_info_out = output_dir / "all.txt"
        else:
            scene_info_out = scene_output_dir / f"{scene_id}.txt"

        with open(scene_info_out, "w") as f:
            f.write("\n".join(lines))

        json_results = write_json_results(confmats, semantic_classes)
        with open(scene_info_out.with_suffix(".json"), "w") as f:
            json.dump(json_results, f, indent=4)

        print(f"Evaluation of {scene_id} done in: {time.time() - start:.2f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("preds_dir", help="Path to predictions directory")
    parser.add_argument(
        "--config",
        default="configs/scannetpp_eval/eval_semantic.yml",
        help="Path to config file",
    )
    parser.add_argument(
        "--scene-id",
        type=str,
        help="The scene id if only one scene should be evaluated",
    )

    args = parser.parse_args()

    main(
        Path(args.preds_dir),
        Path(args.config),
        args.scene_id,
    )
