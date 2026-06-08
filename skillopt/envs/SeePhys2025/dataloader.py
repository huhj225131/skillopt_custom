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


def _normalize_item(row: dict[str, Any], *, fallback_id: int, cache_dir: Path) -> dict[str, Any]:
    item_id = str(row.get("index") or fallback_id).strip()
    question = _normalize_text(row.get("question"))
    answer = _normalize_text(row.get("answer"))
    image_paths = _materialize_images(row.get("images"), item_id=item_id, cache_dir=cache_dir)
    metadata = {
        "reasoning": _normalize_text(row.get("reasoning")),
        "sig_figs": _normalize_text(row.get("sig_figs")),
        "level": row.get("level"),
        "subject": _normalize_text(row.get("subject")),
        "language": _normalize_text(row.get("language")),
        "img_category": _normalize_text(row.get("img_category")),
        "vision_relevance": _normalize_text(row.get("vision_relevance")),
        "caption": _normalize_text(row.get("caption")),
    }
    task_type = metadata["subject"] or metadata["img_category"] or metadata["vision_relevance"] or "seephys"
    return {
        "id": item_id,
        "index": row.get("index", fallback_id),
        "question": question,
        "answer": answer,
        "answers": [answer] if answer else [],
        "ground_truth": answer,
        "image_paths": image_paths,
        "images": image_paths,
        "task_type": task_type,
        "subtask": task_type,
        "metadata": metadata,
    }


class SeePhys2025DataLoader(BaseDataLoader):
    def __init__(
        self,
        data_path: str = "SeePhys_data",
        split_mode: str = "ratio",
        split_ratio: str = "2:1:7",
        split_seed: int = 42,
        seed: int = 42,
        limit: int = 0,
        image_cache_dir: str = "",
        **kwargs,
    ) -> None:
        self.data_path = data_path
        self.split_mode = split_mode
        self.split_ratio = split_ratio
        self.split_seed = int(split_seed)
        self.seed = seed
        self.limit = limit
        self.image_cache_dir = image_cache_dir
        self.train_items: list[dict[str, Any]] = []
        self.val_items: list[dict[str, Any]] = []
        self.test_items: list[dict[str, Any]] = []
        self._task_types: list[str] = []

    def _load_all_items(self) -> list[dict[str, Any]]:
        dataset = load_from_disk(self.data_path)
        cache_dir = Path(self.image_cache_dir or Path(self.data_path) / "_materialized_images")
        split_datasets: list[Dataset]
        if isinstance(dataset, DatasetDict):
            split_datasets = [dataset[name] for name in dataset.keys()]
        elif isinstance(dataset, Dataset):
            split_datasets = [dataset]
        else:
            raise TypeError(f"Unsupported dataset type: {type(dataset).__name__}")

        items: list[dict[str, Any]] = []
        for split_dataset in split_datasets:
            formatted = split_dataset.cast_column("images", Sequence(Image(decode=False)))
            for row_idx in range(len(formatted)):
                row = formatted[row_idx]
                items.append(_normalize_item(row, fallback_id=len(items), cache_dir=cache_dir))
        return items

    def _split_items(self, items: list[dict[str, Any]]) -> None:
        if self.limit > 0:
            items = items[: self.limit]
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
        raw_counts = [total * part / denom for part in (train_ratio, val_ratio, test_ratio)]
        counts = [int(value) for value in raw_counts]
        remaining = total - sum(counts)
        order = sorted(
            range(3),
            key=lambda idx: (raw_counts[idx] - counts[idx], [train_ratio, val_ratio, test_ratio][idx]),
            reverse=True,
        )
        for idx in order[:remaining]:
            counts[idx] += 1
        if total > 0 and counts[0] == 0:
            donor_candidates = [idx for idx in (1, 2) if counts[idx] > 0]
            if donor_candidates:
                donor = max(donor_candidates, key=lambda idx: counts[idx])
                counts[donor] -= 1
                counts[0] += 1
        n_train, n_val, n_test = counts
        if n_train + n_val + n_test < total:
            n_test += total - (n_train + n_val + n_test)
        self.train_items = shuffled[:n_train]
        self.val_items = shuffled[n_train:n_train + n_val]
        self.test_items = shuffled[n_train + n_val:]

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        self.data_path = cfg.get("data_path", self.data_path)
        self.split_mode = cfg.get("split_mode", self.split_mode)
        self.split_ratio = cfg.get("split_ratio", self.split_ratio)
        self.split_seed = int(cfg.get("split_seed", self.split_seed) or self.split_seed)
        self.limit = int(cfg.get("limit", self.limit) or self.limit)
        self.image_cache_dir = cfg.get("image_cache_dir", self.image_cache_dir)

        all_items = self._load_all_items()
        self._split_items(all_items)

        seen: set[str] = set()
        task_types: list[str] = []
        for item in self.train_items + self.val_items + self.test_items:
            task_type = str(item.get("task_type") or "seephys").strip() or "seephys"
            if task_type not in seen:
                seen.add(task_type)
                task_types.append(task_type)
        self._task_types = task_types or ["seephys"]

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