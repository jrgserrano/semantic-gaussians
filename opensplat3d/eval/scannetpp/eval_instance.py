# Call instance evaluation with support for class-agnostic evaluationa, do it also per scene and store the results in a file

import json
import os
import time
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
from scannetpp.common.file_io import (
    load_json,
    load_yaml_munch,
    read_txt_list,
)
from scannetpp.common.scene_release import ScannetppScene_Release
from scannetpp.common.utils.rle import rle_decode
from scannetpp.semantic.eval.eval_instance import (
    compute_averages,
    evaluate_matches,
    print_results,
    verify_pred_files,
)
from scannetpp.semantic.utils import instance_utils
from scannetpp.semantic.utils.instance_utils import Instance
from tqdm import tqdm


def get_instances(
    ids, valid_class_ids, valid_class_labels, id2label, class_agnostic: bool = False
):
    instances = {}
    # each class name
    for label in valid_class_labels:
        # instances in this class
        instances[label] = []
    # unique instance IDs
    instance_ids = np.unique(ids)
    # ignore instance ID 0!
    for id in instance_ids:
        if id == 0:
            continue
        # create new instance object
        inst = Instance(ids, id)
        if inst.label_id in valid_class_ids:
            if class_agnostic:
                # remove semantic label
                inst.label_id = 0
            instances[id2label[inst.label_id]].append(inst.to_dict())
    # list of instances as dicts
    return instances


def assign_instances_for_scan(
    pred_file,
    gt_file,
    preds_dir,
    ignore_mask,
    label_info,
    eval_opts,
    cg_label_info=None,
):
    try:
        # read pred_file_path, label, conf for each instance
        # dict: pred path -> dict: label_info, conf
        pred_info = instance_utils.read_instance_prediction_file(pred_file, preds_dir)
    except Exception as e:
        raise ValueError(f"Error reading {pred_file}: {e}")

    # read the GT file as an array of ints
    # single array of GT
    gt_ids = instance_utils.load_ids(gt_file)

    # dont eval on masked regions
    # keep all preds and gt except masked regions
    vtx_ndx = np.arange(len(gt_ids))
    # vertices to keep
    keep_vtx = ~np.isin(vtx_ndx, ignore_mask)

    # keep only unmasked GT
    gt_ids = gt_ids[keep_vtx]

    # get gt instances
    # GT as instance objects
    # dict: class label -> list of instances as dict with
    #        instance_id (1000sem+inst), label_id (semantic label), vert_count, med_dist (-1), dist_conf (0)
    orig_label_info = label_info
    if cg_label_info is not None:
        label_info = cg_label_info

    gt_instances = get_instances(
        gt_ids,
        orig_label_info.valid_class_ids,
        label_info.class_labels,
        label_info.id_to_label,
        class_agnostic=cg_label_info is not None,
    )

    assert sum([len(x) for x in gt_instances.values()]), (
        "No valid instances found in GT"
    )

    # associate
    # for each GT instance, all the matched predictions = list
    gt2pred = deepcopy(gt_instances)
    # for each GT class
    for label in gt2pred:
        # each instance in that class
        for gt in gt2pred[label]:
            # matched preds for this gt instance are empty
            gt["matched_pred"] = []

    pred2gt = {}
    # for each class
    for label in label_info.class_labels:
        # matched gts for this class are empty
        pred2gt[label] = []

    num_pred_instances = 0
    # mask of invalid semantic labels
    bool_void = np.logical_not(np.isin(gt_ids // 1000, orig_label_info.valid_class_ids))

    # go thru all prediction masks
    for pred_mask_file in pred_info:
        # get the sem label and conf for pred
        label_id = int(pred_info[pred_mask_file]["label_id"])
        conf = pred_info[pred_mask_file]["conf"]

        if label_id not in label_info.id_to_label:
            continue

        label_name = label_info.id_to_label[label_id]
        # load mask from RLE JSON
        pred_mask = rle_decode(load_json(pred_mask_file))
        # keep only unmasked preds
        pred_mask = pred_mask[keep_vtx]
        assert len(pred_mask) == len(gt_ids), (
            f"Wrong number of lines in {pred_mask_file}: {len(pred_mask)} vs #mesh vertices {len(gt_ids)}, please check and/or re-download the mesh"
        )

        # convert to binary
        num = np.count_nonzero(pred_mask)

        # dont have enough vertices with indices
        if num < eval_opts.min_region_sizes[0]:
            continue  # skip if empty

        # create a new instance dict
        pred_instance = {}
        pred_instance["filename"] = pred_mask_file
        # assign a new id, keep incrementing
        pred_instance["pred_id"] = num_pred_instances
        num_pred_instances += 1
        # semantic class ID
        pred_instance["label_id"] = label_id
        # num vertices in this pred instance
        pred_instance["vert_count"] = num
        # predicted confidence
        pred_instance["confidence"] = conf
        # places where the semantic class is invalid
        pred_instance["void_intersection"] = np.count_nonzero(
            np.logical_and(bool_void, pred_mask)
        )

        # matched gt instances
        matched_gt = []
        # go thru all gt instances with matching label
        for gt_num, gt_inst in enumerate(gt2pred[label_name]):
            # intersection of gt mask and pred mask
            gt_mask = gt_ids == gt_inst["instance_id"]
            intersection = np.count_nonzero(np.logical_and(gt_mask, pred_mask))
            if intersection > 0:
                gt_copy = gt_inst.copy()
                pred_copy = pred_instance.copy()
                gt_copy["intersection"] = intersection
                pred_copy["intersection"] = intersection
                # add to list of all matched GT for this pred
                matched_gt.append(gt_copy)
                # add pred as match to GT
                gt2pred[label_name][gt_num]["matched_pred"].append(pred_copy)
        # all matched GT for this pred
        pred_instance["matched_gt"] = matched_gt
        pred2gt[label_name].append(pred_instance)

    # pred2gt: pred instances + matched GT for each pred
    # gt2pred: GT instances + matched pred for each GT
    return gt2pred, pred2gt


def evaluate(
    pred_files,
    gt_files,
    preds_dir,
    data_root,
    label_info,
    eval_opts,
    cg_label_info=None,
):
    print(f"Evaluating {len(pred_files)} scans")
    matches = {}

    for scene_ndx, pred_file in enumerate(tqdm(pred_files, desc="assign_pred_scene")):
        # get the scene id
        scene_id = pred_file.stem

        # create scene object to get the mesh mask
        scene = ScannetppScene_Release(scene_id, data_root=data_root)

        # vertices to ignore for eval
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, append=True)
            ignore_mask = np.loadtxt(scene.scan_mesh_mask_path, dtype=np.int32)

        matches_key = os.path.abspath(gt_files[scene_ndx])

        # assign gt to predictions
        # gt = 1 file per scene
        # pred = 1 file per scene + 1 file per instance
        # output:
        # pred2gt: pred instances as dicts + matched GT for each pred
        # gt2pred: GT instances as dict + matched pred for each GT
        gt2pred, pred2gt = assign_instances_for_scan(
            pred_file,
            gt_files[scene_ndx],
            preds_dir,
            ignore_mask,
            label_info,
            eval_opts,
            cg_label_info,
        )
        # for each scene, gt2pred and pred2gt
        matches[matches_key] = {"gt": gt2pred, "pred": pred2gt}

    if cg_label_info is not None:
        label_info = cg_label_info

    # get scores
    # does the greedy assignment
    # evaluate all scenes together
    ap_scores = evaluate_matches(matches, label_info, eval_opts)
    avgs = compute_averages(ap_scores, label_info, eval_opts)

    # print(preds_dir)
    # preds_dir = Path(preds_dir)
    # np.save(preds_dir.parent / f"{preds_dir.name}_ap_scores.npy", ap_scores)

    return avgs


def eval_instance(
    scene_ids,
    preds_dir,
    gt_dir,
    data_root,
    label_info,
    eval_opts,
    check_pred_files=False,
    cg_label_info=None,
):
    preds_dir = Path(preds_dir)
    gt_dir = Path(gt_dir)

    pred_files, gt_files = [], []

    # check if all pred and gt files exist
    for scene_id in scene_ids:
        pred_file = preds_dir / f"{scene_id}.txt"
        gt_file = gt_dir / f"{scene_id}.txt"

        if not os.path.isfile(pred_file):
            raise FileNotFoundError(f"Prediction file {pred_file} does not exist")
        if not os.path.isfile(gt_file):
            raise FileNotFoundError(f"GT file {gt_file} does not exist")

        pred_files.append(pred_file)
        gt_files.append(gt_file)

    if check_pred_files:
        label_info_ = label_info
        if cg_label_info is not None:
            label_info_ = cg_label_info
        verify_pred_files(pred_files, label_info_)

    # evaluate
    results = evaluate(
        pred_files,
        gt_files,
        preds_dir,
        data_root,
        label_info,
        eval_opts,
        cg_label_info,
    )

    return results


def write_results(avgs, label_info):
    sep = ""
    col1 = ":"
    lineLen = 64

    lines = []

    lines.append("")
    lines.append("#" * lineLen)
    line = ""
    line += "{:<15}".format("what") + sep + col1
    line += "{:>15}".format("AP") + sep
    line += "{:>15}".format("AP_50%") + sep
    line += "{:>15}".format("AP_25%") + sep
    lines.append(line)
    lines.append("#" * lineLen)

    for label_name in label_info.class_labels:
        ap_avg = avgs["classes"][label_name]["ap"]
        ap_50o = avgs["classes"][label_name]["ap50%"]
        ap_25o = avgs["classes"][label_name]["ap25%"]
        line = "{:<15}".format(label_name) + sep + col1
        line += sep + "{:>15.3f}".format(ap_avg) + sep
        line += sep + "{:>15.3f}".format(ap_50o) + sep
        line += sep + "{:>15.3f}".format(ap_25o) + sep
        lines.append(line)

    all_ap_avg = avgs["all_ap"]
    all_ap_50o = avgs["all_ap_50%"]
    all_ap_25o = avgs["all_ap_25%"]

    lines.append("-" * lineLen)
    line = "{:<15}".format("average") + sep + col1
    line += "{:>15.3f}".format(all_ap_avg) + sep
    line += "{:>15.3f}".format(all_ap_50o) + sep
    line += "{:>15.3f}".format(all_ap_25o) + sep
    lines.append(line)
    lines.append("")
    return lines


def write_json_results(avgs, label_info):
    results = {}
    results["classes"] = {}
    for label_name in label_info.class_labels:
        ap_avg = avgs["classes"][label_name]["ap"]
        ap_50o = avgs["classes"][label_name]["ap50%"]
        ap_25o = avgs["classes"][label_name]["ap25%"]
        results["classes"][label_name] = {
            "ap": ap_avg,
            "ap50": ap_50o,
            "ap25": ap_25o,
        }

    results["all"] = {}
    results["all"]["ap"] = avgs["all_ap"]
    results["all"]["ap50"] = avgs["all_ap_50%"]
    results["all"]["ap25"] = avgs["all_ap_25%"]

    return results


def main(
    preds_dir: Path,
    config_file: Path,
    scene_id: str | None,
    per_scene: bool,
    class_agnostic: bool,
):
    cfg = load_yaml_munch(config_file)
    cfg.scene_list_file = os.path.expandvars(cfg.scene_list_file)
    cfg.gt_dir = os.path.expandvars(cfg.gt_dir)
    cfg.data_root = os.path.expandvars(cfg.data_root)
    cfg.semantic_classes_file = os.path.expandvars(cfg.semantic_classes_file)
    cfg.instance_classes_file = os.path.expandvars(cfg.instance_classes_file)

    eval_name = f"instance{'_agnostic' if class_agnostic else ''}"

    is_eval_run = (preds_dir.parent.parent / "scenes").exists()
    if not is_eval_run and scene_id is None:
        # {exp_dir}/eval_predictions/semantic/{pred_dir}
        pred_files = list((preds_dir / eval_name).glob("*.txt"))
        assert len(pred_files) == 1, (
            f"Expected exactly one prediction file in {preds_dir / eval_name}, found {len(pred_files)}"
        )
        scene_id = pred_files[0].stem
    assert is_eval_run or scene_id is not None, (
        "Expected eval run structure with 'scenes' directory two levels up or a single scene_id."
    )
    output_dir = preds_dir.parent.parent / "eval_results" / preds_dir.name

    preds_dir = preds_dir / eval_name
    output_dir = output_dir / eval_name

    cfg.preds_dir = str(preds_dir)

    print(f"Using predictions from {preds_dir}")
    print(f"Output directory: {output_dir}")

    all_scene_ids = sorted(read_txt_list(cfg.scene_list_file))
    if scene_id is not None:
        assert scene_id in all_scene_ids, (
            f"The provided scene id '{scene_id}' is not in the list."
        )
        all_scene_ids = [scene_id]
    semantic_class_list = read_txt_list(cfg.semantic_classes_file)
    instance_class_list = read_txt_list(cfg.instance_classes_file)

    # labels to evaluate on, label-id mappings
    label_info = instance_utils.get_label_info(semantic_class_list, instance_class_list)

    # NOTE: class agnostic evaluation! Only one class with id 0!
    cg_label_info = None
    if class_agnostic:
        cg_label_info = instance_utils.get_label_info(["void"], ["void"])

    orig_label_info = label_info
    if cg_label_info is not None:
        label_info = cg_label_info

    # evaluation parameters, can be customized
    eval_opts = instance_utils.Instance_Eval_Opts()

    scene_output_dir = output_dir / "scenes"

    eval_scene_list = [all_scene_ids]
    if per_scene:
        eval_scene_list = [[x] for x in all_scene_ids] + eval_scene_list

    if len(eval_scene_list) > 1:
        scene_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    for i, scene_ids in enumerate(eval_scene_list):
        start = time.time()
        is_all = i == len(eval_scene_list) - 1
        scene_id = scene_ids[0] if not is_all else "all"

        all_results = eval_instance(
            scene_ids,
            cfg.preds_dir,
            cfg.gt_dir,
            cfg.data_root,
            orig_label_info,
            eval_opts,
            cfg.check_pred_files,
            cg_label_info,
        )

        lines = write_results(all_results, label_info)
        if is_all:
            print_results(all_results, label_info)
            scene_info_out = output_dir / "all.txt"
        else:
            scene_info_out = scene_output_dir / f"{scene_id}.txt"

        with open(scene_info_out, "w") as f:
            f.write("\n".join(lines))

        json_results = write_json_results(all_results, label_info)
        with open(scene_info_out.with_suffix(".json"), "w") as f:
            json.dump(json_results, f, indent=4)

        print(f"Evaluation of {scene_id} done in: {time.time() - start:.2f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("preds_dir", help="Path to predictions directory")
    parser.add_argument(
        "--config",
        default="configs/scannetpp_eval/eval_instance.yml",
        help="Path to config file",
    )
    parser.add_argument(
        "--scene-id",
        type=str,
        help="The scene id if only one scene should be evaluated",
    )
    parser.add_argument("--per-scene", action="store_true", help="Evaluate per scene")
    parser.add_argument(
        "--class-agnostic", action="store_true", help="Class agnostic evaluation"
    )

    args = parser.parse_args()

    main(
        Path(args.preds_dir),
        Path(args.config),
        args.scene_id,
        args.per_scene,
        args.class_agnostic,
    )
