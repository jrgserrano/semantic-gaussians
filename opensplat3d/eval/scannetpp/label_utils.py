"""Adapted from ScanNet++"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class LabelInfo:
    all_class_labels: list[str]
    all_class_ids: list[int]

    ignore_classes: list[int]
    valid_class_ids: list[int]
    class_labels: list[str]

    id_to_label: dict[int, str]
    label_to_id: dict[str, int]


def get_label_info(
    semantic_class_list: list[str], instance_class_list: list[str]
) -> LabelInfo:
    label_info = LabelInfo(
        all_class_labels=semantic_class_list,
        all_class_ids=list(range(len(semantic_class_list))),
        ignore_classes=[],
        valid_class_ids=[],
        class_labels=[],
        id_to_label={},
        label_to_id={},
    )
    # indices of semantic classes not present in instance class list
    label_info.ignore_classes = [
        i
        for i in range(len(label_info.all_class_labels))
        if label_info.all_class_labels[i] not in instance_class_list
    ]
    # ids of instance classes (all semantic classes - ignored classes)
    label_info.class_labels = [
        label_info.all_class_labels[i]
        for i in label_info.all_class_ids
        if i not in label_info.ignore_classes
    ]
    # ids of instance classes
    label_info.valid_class_ids = [
        i for i in label_info.all_class_ids if i not in label_info.ignore_classes
    ]

    for i in range(len(label_info.valid_class_ids)):
        # class id -> class name
        label_info.id_to_label[label_info.valid_class_ids[i]] = label_info.class_labels[
            i
        ]
        # class name -> class id
        label_info.label_to_id[label_info.class_labels[i]] = label_info.valid_class_ids[
            i
        ]

    return label_info


def load_label_infos(root_path: Path) -> tuple[LabelInfo, LabelInfo]:
    with open(root_path / "metadata" / "semantic_benchmark" / "top100.txt") as f:
        top100_labels = f.read().strip().splitlines()

    with open(
        root_path / "metadata" / "semantic_benchmark" / "top100_instance.txt"
    ) as f:
        top100_instance_labels = f.read().strip().splitlines()

    return (
        get_label_info(top100_labels, top100_labels),
        get_label_info(top100_labels, top100_instance_labels),
    )
