"""Fixed CSV indices and the original aggregate-HDF5 bag loading semantics."""

import csv
from pathlib import Path

import h5py
import numpy as np


EXPECTED_COUNTS = {"camelyon": (242, 27, 129), "bracs": (389, 62, 86),
                   "nsclc": (885, 46, 120)}
SPLITS = ("train", "val", "test")


def read_split_csv(path, expected_counts):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Supply the author's fixed split CSV: {path}")
    names = {split: [] for split in SPLITS}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        split_column = "split" if "split" in fields else "split_info"
        long_format = "slide_id" in fields and split_column in fields
        wide_format = set(SPLITS).issubset(fields)
        if not long_format and not wide_format:
            raise ValueError("CSV requires slide_id,split (or split_info), or train,val,test columns")
        for line, row in enumerate(reader, start=2):
            if long_format:
                slide_id = (row.get("slide_id") or "").strip()
                split = (row.get(split_column) or "").strip()
                if not slide_id or split not in names:
                    raise ValueError(f"Invalid slide_id/split at CSV line {line}")
                names[split].append(slide_id)
            else:
                for split in SPLITS:
                    slide_id = (row.get(split) or "").strip()
                    if slide_id:
                        names[split].append(slide_id)
    seen = set()
    for split, count in zip(SPLITS, expected_counts):
        if len(names[split]) != count:
            raise ValueError(f"{split}: expected {count} slides, got {len(names[split])}")
        for slide_id in names[split]:
            if slide_id in seen:
                raise ValueError(f"Duplicate/overlapping slide_id: {slide_id}")
            seen.add(slide_id)
    return names


class HDF5_feat_dataset2:
    # Preserve data_dict/data_names, return values and eager loading from datasets.py.
    def __init__(self, data_dict, data_names):
        self.data_dict = data_dict
        self.data_names = data_names

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, index):
        return self.data_dict[self.data_names[index]]


def build_datasets(file_path, split_path, conf):
    names = read_split_csv(split_path, EXPECTED_COUNTS[conf.dataset])
    result = []
    with h5py.File(file_path, "r") as h5_data:
        for split_name in SPLITS:
            data = {}
            split_names = names[split_name]
            if conf.dataset == "bracs":
                # Original BRACS iterates HDF5 keys, rather than metadata CSV rows.
                selected = set(split_names)
                missing = selected.difference(h5_data.keys())
                if missing:
                    raise ValueError(f"Missing HDF5 slide: {sorted(missing)[0]}")
                split_names = [name for name in h5_data.keys() if name in selected]
            for name in split_names:
                if name not in h5_data:
                    raise ValueError(f"Missing HDF5 slide: {name}")
                slide = h5_data[name]
                raw_label = slide.attrs["label"]
                if np.asarray(raw_label).shape != () or not np.isfinite(raw_label) or int(raw_label) != raw_label:
                    raise ValueError(f"Invalid scalar label for {name}: {raw_label!r}")
                label = int(raw_label)
                if conf.dataset == "bracs":
                    transfer = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
                    if label not in transfer:
                        raise ValueError(f"BRACS requires original seven-type labels: {name}")
                    label = transfer[label]
                if not 0 <= label < conf.n_class:
                    raise ValueError(f"Label out of range for {name}: {label}")
                feat = slide["feat"][:]
                coords = slide["coords"][:]
                if feat.ndim != 2 or feat.shape[0] == 0 or feat.shape[1] != conf.D_feat or not np.isfinite(feat).all():
                    raise ValueError(f"Expected finite, nonempty [N, {conf.D_feat}] features: {name}")
                if coords.shape != (feat.shape[0], 2):
                    raise ValueError(f"Expected [N, 2] patch coordinates: {name}")
                data[name] = {"input": feat, "coords": coords, "label": label}
            result.append(HDF5_feat_dataset2(data, split_names))
    print("Loaded fixed splits:", {split: len(ds) for split, ds in zip(SPLITS, result)})
    return tuple(result)
