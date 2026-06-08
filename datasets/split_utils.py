import json
import os
from typing import Dict, List, Tuple


def load_split_paths(
    split_file: str,
    data_root: str,
    class_to_idx: Dict[str, int],
    strict_paths: bool = True,
) -> Tuple[List[str], List[int]]:
    with open(split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    rel_paths = split_data.get("image_paths", [])
    if not rel_paths:
        raise ValueError(f"No image_paths found in split file: {split_file}")

    image_paths: List[str] = []
    labels: List[int] = []
    missing_paths: List[str] = []
    unknown_classes = set()

    for rel_path in rel_paths:
        normalized_rel = rel_path.replace("\\", "/")
        class_name = normalized_rel.split("/", 1)[0]
        if class_name not in class_to_idx:
            unknown_classes.add(class_name)
            continue

        full_path = os.path.join(data_root, *normalized_rel.split("/"))
        if strict_paths and not os.path.isfile(full_path):
            missing_paths.append(full_path)

        image_paths.append(full_path)
        labels.append(class_to_idx[class_name])

    if unknown_classes:
        known_preview = ", ".join(list(class_to_idx.keys())[:8])
        unknown_preview = ", ".join(sorted(unknown_classes)[:8])
        raise ValueError(
            f"Split file {split_file} contains classes not found under {data_root}: "
            f"{unknown_preview}. Known class preview: {known_preview}"
        )

    if missing_paths:
        examples = "\n".join(missing_paths[:5])
        raise FileNotFoundError(
            f"{len(missing_paths)} paths from {split_file} do not exist under {data_root}. "
            f"First missing paths:\n{examples}"
        )

    return image_paths, labels
