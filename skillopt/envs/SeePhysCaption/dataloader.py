from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, Image, Sequence, load_from_disk

from skillopt.datasets.base import BatchSpec, BaseDataLoader


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_filename(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return text.strip("_") or "item"


def _materialize_images(raw_images: Any, *, item_id: str, cache_dir: Path) -> list[str]:
    images = raw_images if isinstance(raw_images, list) else [raw_images]
    cache_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[str] = []
    base_name = _safe_filename(item_id)

    for index, image in enumerate(images):
        if not image:
            continue
        if isinstance(image, dict):
            image_bytes = image.get("bytes")
            image_path = image.get("path")
            if image_bytes:
                suffix = Path(str(image_path) or f"{base_name}_{index}.png").suffix or ".png"
                output_path = cache_dir / f"{base_name}_{index}{suffix}"
                output_path.write_bytes(image_bytes)
                image_paths.append(str(output_path))
                continue
            if image_path:
                image_paths.append(str(image_path))
                continue
        if hasattr(image, "save"):
            output_path = cache_dir / f"{base_name}_{index}.png"
            image.save(output_path)
            image_paths.append(str(output_path))

    return image_paths


def _normalize_item(row: dict[str, Any], ground_truth: str, *, fallback_id: int, cache_dir: Path) -> dict[str, Any]:
    item_id = str(row.get("question_id", row.get("index", fallback_id))).strip()
    input_text = _normalize_text(row.get("problem"))
    image_paths = _materialize_images(row.get("images"), item_id=item_id, cache_dir=cache_dir)
    
    return {
        "id": item_id,
        "index": row.get("idx", row.get("index", fallback_id)),
        "question": input_text,
        "input_text": input_text,
        "answer": ground_truth,
        "answers": [ground_truth] if ground_truth else [],
        "ground_truth": ground_truth,
        "image_paths": image_paths,
        "images": image_paths,
        "task_type": "seephys_caption",
        "subtask": "seephys_caption",
    }


class SeePhysCaptionDataLoader(BaseDataLoader):
    def __init__(
        self,
        base_data_path: str = "/media/hung/DATA/SkillOpt/SeePhys_2026_data",
        target_level: str = "level1",
        input_levels: str = "level2,level3,level4",
        split_mode: str = "ratio",
        split_ratio: str = "2:1:7",
        split_seed: int = 42,
        seed: int = 42,
        limit: int = 0,
        image_cache_dir: str = "",
        **kwargs,
    ) -> None:
        self.base_data_path = Path(base_data_path)
        self.target_level = target_level
        self.input_levels = [lvl.strip() for lvl in input_levels.split(",") if lvl.strip()]
        self.split_mode = split_mode
        self.split_ratio = split_ratio
        self.split_seed = int(split_seed)
        self.seed = seed
        self.limit = limit
        self.image_cache_dir = image_cache_dir
        self.train_items: list[dict[str, Any]] = []
        self.val_items: list[dict[str, Any]] = []
        self.test_items: list[dict[str, Any]] = []
        self._task_types: list[str] = ["seephys_caption"]

    def _load_level_data(self, level: str) -> list[dict[str, Any]]:
        path = self.base_data_path / level
        dataset = load_from_disk(str(path))
        split_datasets: list[Dataset] = []
        if isinstance(dataset, DatasetDict):
            split_datasets = [dataset[name] for name in dataset.keys()]
        elif isinstance(dataset, Dataset):
            split_datasets = [dataset]
            
        items = []
        for split_dataset in split_datasets:
            # We don't cast images for the target level to save memory/time
            if level in self.input_levels:
                split_dataset = split_dataset.cast_column("images", Sequence(Image(decode=False)))
            for i in range(len(split_dataset)):
                items.append(split_dataset[i])
        return items

    def _load_all_items(self) -> list[dict[str, Any]]:
        # Load Level 1 (Target)
        target_items = self._load_level_data(self.target_level)
        # Create mapping id -> problem text
        target_map = {str(item.get("question_id", item.get("idx"))): _normalize_text(item.get("problem")) for item in target_items}

        # Load Level X (Input) - can be multiple levels
        input_items = []
        for level in self.input_levels:
            input_items.extend(self._load_level_data(level))
        
        cache_dir = Path(self.image_cache_dir or self.base_data_path / "_materialized_images")
        
        items: list[dict[str, Any]] = []
        for i, row in enumerate(input_items):
            item_id = str(row.get("question_id", row.get("idx")))
            ground_truth = target_map.get(item_id, "")
            
            # Skip if no ground truth matches
            if not ground_truth:
                continue
                
            items.append(_normalize_item(row, ground_truth, fallback_id=len(items), cache_dir=cache_dir))
            
            if self.limit > 0 and len(items) >= self.limit:
                break
                
        return items

    def _split_items(self, items: list[dict[str, Any]]) -> None:
        if not items:
            self.train_items = []
            self.val_items = []
            self.test_items = []
            return
            
        rng = random.Random(self.split_seed)
        shuffled = list(items)
        rng.shuffle(shuffled)
        total = len(shuffled)
        
        train_ratio, val_ratio, test_ratio = (int(part) for part in str(self.split_ratio).split(":"))
        denom = train_ratio + val_ratio + test_ratio
        
        n_train = int(total * train_ratio / denom)
        n_val = int(total * val_ratio / denom)
        n_test = total - n_train - n_val
        
        self.train_items = shuffled[:n_train]
        self.val_items = shuffled[n_train:n_train + n_val]
        self.test_items = shuffled[n_train + n_val:]

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        self.base_data_path = Path(cfg.get("base_data_path", self.base_data_path))
        self.target_level = cfg.get("target_level", self.target_level)
        raw_levels = cfg.get("input_levels", "")
        if raw_levels:
            self.input_levels = [lvl.strip() for lvl in raw_levels.split(",") if lvl.strip()]
        self.split_ratio = cfg.get("split_ratio", self.split_ratio)
        self.split_seed = int(cfg.get("split_seed", self.split_seed) or self.split_seed)
        self.limit = int(cfg.get("limit", self.limit) or self.limit)
        self.image_cache_dir = cfg.get("image_cache_dir", self.image_cache_dir)

        all_items = self._load_all_items()
        self._split_items(all_items)

    def get_task_types(self) -> list[str]:
        return list(self._task_types)

    def get_train_size(self) -> int | None:
        return len(self.train_items)

    def _sample_items(self, pool: list[dict[str, Any]], batch_size: int, seed: int) -> list[dict[str, Any]]:
        if not pool:
            return []
        rng = random.Random(seed)
        if batch_size <= len(pool):
            indices = list(range(len(pool)))
            rng.shuffle(indices)
            return [pool[idx] for idx in indices[:batch_size]]
        return [rng.choice(pool) for _ in range(batch_size)]

    def build_train_batch(self, batch_size: int, seed: int, **kwargs) -> BatchSpec:
        payload = self._sample_items(self.train_items, batch_size, seed)
        return BatchSpec(phase="train", split="train", seed=seed, batch_size=batch_size, payload=payload)

    def build_eval_batch(self, env_num: int, split: str, seed: int, **kwargs) -> BatchSpec:
        normalized_split = str(split or "test").strip().lower()
        if normalized_split in {"valid_seen", "selection", "val", "valid"}:
            pool = self.val_items or self.test_items or self.train_items
            split_name = "val"
        else:
            pool = self.test_items or self.val_items or self.train_items
            split_name = "test"
        payload = self._sample_items(pool, env_num, seed)
        return BatchSpec(phase="eval", split=split_name, seed=seed, batch_size=env_num, payload=payload)
