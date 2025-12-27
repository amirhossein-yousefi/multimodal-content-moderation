"""
Data preparation utilities for downloading and processing datasets.
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict

import pandas as pd
from tqdm import tqdm
from huggingface_hub import snapshot_download


def ensure_dir(p: Path) -> None:
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)


def write_class_names(out_dir: Path, class_names: List[str]) -> None:
    """Write class names to a text file."""
    with open(out_dir / "class_names.txt", "w", encoding="utf-8") as f:
        for c in class_names:
            f.write(c + "\n")


def download_and_prepare_hateful_memes(out_root: Path) -> None:
    """
    Download and prepare the Hateful Memes dataset.
    
    Source: Hugging Face dataset mirror with images + jsonl
    Structure: img/, train.jsonl, dev_seen.jsonl, test_seen.jsonl
    
    Args:
        out_root: Root directory for output data.
    """
    print("==> Downloading Hateful Memes (HF mirror) ...")
    repo_id = "neuralcatcher/hateful_memes"
    local_repo = Path(snapshot_download(repo_id=repo_id, repo_type="dataset"))
    print(f"Downloaded snapshot to: {local_repo}")

    # Prepare output structure
    out_dir = out_root / "hateful_memes"
    images_out = out_dir / "images"
    ensure_dir(images_out)

    # Copy license if present
    for licename in ("LICENSE", "LICENSE.txt", "license.txt"):
        lic = local_repo / licename
        if lic.exists():
            shutil.copy2(lic, out_dir / "LICENSE.txt")
            break

    # Copy images
    src_img_dir = local_repo / "img"
    if not src_img_dir.exists():
        raise FileNotFoundError(f"Expected 'img' folder inside {local_repo}, but not found.")
    
    print("==> Copying images ...")
    if images_out.exists() and any(images_out.iterdir()):
        print("Images folder already populated; skipping copy.")
    else:
        shutil.copytree(src_img_dir, images_out, dirs_exist_ok=True)

    def _read_jsonl(path: Path) -> List[Dict]:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
        return rows

    # Build splits
    split_map = {
        "train.jsonl": "train.csv",
        "dev_seen.jsonl": "val.csv",
        "test_seen.jsonl": "test.csv",
    }

    for jsonl_name, csv_name in split_map.items():
        jsonl_path = local_repo / jsonl_name
        if not jsonl_path.exists():
            print(f"Warning: {jsonl_name} not found, skipping...")
            continue
            
        rows = _read_jsonl(jsonl_path)
        records = []
        for row in tqdm(rows, desc=f"Processing {jsonl_name}"):
            img_path = row.get("img", "")
            text = row.get("text", "")
            label = row.get("label", 0)
            records.append({
                "text": text,
                "image_path": os.path.basename(img_path),
                "label": int(label)
            })
        
        df = pd.DataFrame(records)
        df.to_csv(out_dir / csv_name, index=False)
        print(f"Saved {csv_name} with {len(df)} samples")

    # Write class names for binary classification
    write_class_names(out_dir, ["hateful"])
    print(f"==> Hateful Memes prepared at: {out_dir}")


def prepare_mmhs150k_from_raw(raw_dir: Path, out_dir: Path) -> None:
    """
    Prepare MMHS150K dataset from raw files.
    
    Args:
        raw_dir: Path to raw MMHS150K data (containing MMHS150K_GT.json).
        out_dir: Output directory for processed data.
    """
    gt_path = raw_dir / "MMHS150K_GT.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    
    # Class mapping from MMHS150K
    class_names = ["racist", "sexist", "homophobe", "religion", "otherhate"]
    
    ensure_dir(out_dir)
    ensure_dir(out_dir / "images")
    
    # Process splits
    splits_dir = raw_dir / "splits"
    for split_name in ["train", "val", "test"]:
        split_file = splits_dir / f"{split_name}_ids.txt"
        if not split_file.exists():
            print(f"Warning: {split_file} not found, skipping...")
            continue
            
        with open(split_file, "r") as f:
            ids = [line.strip() for line in f if line.strip()]
        
        records = []
        for img_id in tqdm(ids, desc=f"Processing {split_name}"):
            if img_id not in gt:
                continue
            entry = gt[img_id]
            text = entry.get("tweet_text", "")
            labels_list = entry.get("labels", [])
            
            # Convert label indices to class names
            active_labels = []
            for idx in labels_list:
                if 0 <= idx < len(class_names):
                    active_labels.append(class_names[idx])
            
            records.append({
                "text": text,
                "image_path": f"{img_id}.jpg",
                "labels": ",".join(active_labels) if active_labels else ""
            })
        
        df = pd.DataFrame(records)
        df.to_csv(out_dir / f"{split_name}.csv", index=False)
        print(f"Saved {split_name}.csv with {len(df)} samples")
    
    # Write class names
    write_class_names(out_dir, class_names)
    print(f"==> MMHS150K prepared at: {out_dir}")
