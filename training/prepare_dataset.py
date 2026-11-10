"""Prepare the added crop datasets for image classification."""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLITS = {"train", "val", "test"}


def images_in(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def clean_name(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value.strip())
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return re.sub(r"_+", "_", value)


def class_name(crop: str, disease: str) -> str:
    disease = "_".join(part.capitalize() for part in clean_name(disease).split("_"))
    if disease.casefold() in {"healthy", "healthy_leaf", "healthy_plant", "healthy_guava"}:
        disease = "Healthy"
    return f"{clean_name(crop)}___{disease}"


def seeded_split(files: list[Path], seed: int) -> tuple[list[Path], list[Path]]:
    shuffled = files.copy()
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 2:
        return shuffled, []
    validation_count = max(1, round(len(shuffled) * 0.2))
    validation_count = min(validation_count, len(shuffled) - 1)
    return shuffled[validation_count:], shuffled[:validation_count]


def source_key(source_root: Path, image: Path) -> str:
    return image.relative_to(source_root).as_posix()


def label_folder_dataset(
    class_root: Path,
    crop: str,
    samples: list[tuple[str, Path, str]],
    seed: int,
    label_mapper=None,
) -> None:
    """Collect class-folder images; sample tuples are split, path, label."""
    for class_dir in sorted(path for path in class_root.iterdir() if path.is_dir()):
        label = label_mapper(class_dir.name) if label_mapper else class_name(crop, class_dir.name)
        files = images_in(class_dir)
        train_files, val_files = seeded_split(files, seed)
        samples.extend(("train", path, label) for path in train_files)
        samples.extend(("val", path, label) for path in val_files)


def split_folder_dataset(
    split_root: Path,
    crop: str,
    samples: list[tuple[str, Path, str]],
    label_mapper=None,
) -> None:
    for split_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        split = split_dir.name.casefold()
        if split not in SPLITS:
            continue
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            label = label_mapper(class_dir.name) if label_mapper else class_name(crop, class_dir.name)
            samples.extend(
                (split, path, label)
                for path in images_in(class_dir)
            )


def stage_file(source: Path, destination: Path, copy_files: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    if copy_files:
        shutil.copy2(source, destination)
        return
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def stage_samples(
    samples: list[tuple[str, Path, str]],
    source_root: Path,
    output_root: Path,
    copy_files: bool,
) -> Counter:
    counts: Counter = Counter()
    for split, image, label in samples:
        if split == "test":
            continue
        try:
            relative_source = source_key(source_root, image)
        except ValueError:
            relative_source = image.relative_to(output_root).as_posix()
        digest = hashlib.sha1(relative_source.encode("utf-8")).hexdigest()[:16]
        destination = output_root / split / label / f"{digest}{image.suffix.lower()}"
        stage_file(image, destination, copy_files)
        counts[(split, label)] += 1
    return counts


def rice_class_names(yaml_path: Path) -> list[str]:
    """Read the simple `names:` list used by the Rice YOLO dataset."""
    names: list[str] = []
    in_names = False
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "names:":
            in_names = True
            continue
        if in_names:
            match = re.match(r"^\s*-\s*['\"]?(.+?)['\"]?\s*$", line)
            if match:
                names.append(match.group(1))
            elif line.strip():
                break
    if not names:
        raise ValueError(f"No class names found in {yaml_path}")
    return names


def rice_label(raw_name: str) -> str:
    if not raw_name.startswith("Rice__"):
        raise ValueError(f"Unexpected Rice class name: {raw_name}")
    disease = raw_name[len("Rice__"):]
    aliases = {
        "BacterialLeafBlight": "Bacterial_Leaf_Blight",
        "BrownSpot": "Brown_Spot",
        "Healthy": "Healthy",
        "Hispa": "Hispa",
        "LeafBlast": "Blast",
        "LeafScald": "Leaf_Scald",
        "LeafSmut": "Leaf_Smut",
        "NarrowBrownLeafSpot": "Narrow_Brown_Leaf_Spot",
        "NeckBlast": "Neck_Blast",
    }
    if disease not in aliases:
        raise ValueError(f"Unknown Rice class in data.yaml: {raw_name}")
    return f"Rice___{aliases[disease]}"


def collect_rice_yolo(
    source_root: Path,
    output_root: Path,
    samples: list[tuple[str, Path, str]],
) -> int:
    rice_root = source_root / "rice"
    names = [rice_label(name) for name in rice_class_names(rice_root / "data.yaml")]
    crop_root = output_root / "rice_crops"
    crop_root.mkdir(parents=True, exist_ok=True)
    crop_count = 0

    for split in ("train", "val"):
        image_dir = rice_root / "images" / split
        label_dir = rice_root / "labels" / split
        for annotation in sorted(label_dir.glob("*.txt")):
            image_path = next(
                (image_dir / f"{annotation.stem}{ext}" for ext in IMAGE_EXTENSIONS
                 if (image_dir / f"{annotation.stem}{ext}").is_file()),
                None,
            )
            if image_path is None:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Cannot decode Rice image: {image_path}")
            height, width = image.shape[:2]
            for box_index, line in enumerate(annotation.read_text(encoding="utf-8").splitlines()):
                values = line.split()
                if not values:
                    continue
                class_index = int(values[0])
                if class_index < 0 or class_index >= len(names):
                    raise ValueError(f"Class index out of range in {annotation}: {class_index}")
                coordinates = list(map(float, values[1:]))
                if len(coordinates) == 4:
                    center_x, center_y, box_width, box_height = coordinates
                    left, top = center_x - box_width / 2, center_y - box_height / 2
                    right, bottom = center_x + box_width / 2, center_y + box_height / 2
                elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
                    xs, ys = coordinates[::2], coordinates[1::2]
                    left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
                else:
                    raise ValueError(f"Invalid YOLO annotation in {annotation}: {line}")
                x0 = max(0, int(left * width))
                y0 = max(0, int(top * height))
                x1 = min(width, int(right * width))
                y1 = min(height, int(bottom * height))
                if x1 - x0 < 16 or y1 - y0 < 16:
                    continue
                label = names[class_index]
                digest = hashlib.sha1(
                    f"{source_key(source_root, image_path)}:{box_index}".encode("utf-8")
                ).hexdigest()[:16]
                crop_path = crop_root / split / label / f"{digest}.jpg"
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                if not crop_path.exists() and not cv2.imwrite(str(crop_path), image[y0:y1, x0:x1]):
                    raise OSError(f"Could not write Rice crop: {crop_path}")
                samples.append((split, crop_path, label))
                crop_count += 1
    return crop_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "organized")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy images instead of hard-linking them.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Replace this script's generated train, val, and rice_crops folders.",
    )
    args = parser.parse_args()
    raw = args.raw.resolve()
    output = args.out.resolve()
    if not raw.is_dir():
        raise SystemExit(f"Raw dataset directory not found: {raw}")
    if args.clean:
        for generated_dir in ("train", "val", "rice_crops"):
            path = output / generated_dir
            if path.is_dir():
                shutil.rmtree(path)

    samples: list[tuple[str, Path, str]] = []

    banana = raw / "Banana Disease Recognition Dataset"
    banana_originals = banana / "Original Images" / "Original Images"
    if banana_originals.is_dir():
        def banana_label(name: str) -> str:
            disease = re.sub(r"^Banana\s+", "", name, flags=re.IGNORECASE)
            disease = re.sub(r"\s+Disease$", "", disease, flags=re.IGNORECASE)
            return class_name("Banana", disease)

        label_folder_dataset(banana_originals, "Banana", samples, args.seed, banana_label)
        print("Banana: using original images only; augmented copies are excluded.")

    cotton = raw / "Cotton Disease"
    if cotton.is_dir():
        def cotton_label(name: str) -> str:
            normalized = clean_name(name).casefold()
            if normalized.startswith("fresh_cotton_"):
                disease = "Healthy_" + normalized.removeprefix("fresh_cotton_")
            elif normalized.startswith("diseased_cotton_"):
                disease = "Diseased_" + normalized.removeprefix("diseased_cotton_")
            else:
                disease = name
            return class_name("Cotton", disease)

        split_folder_dataset(cotton, "Cotton", samples, cotton_label)

    guava = raw / "GuavaDiseaseDataset" / "GuavaDiseaseDataset"
    if guava.is_dir():
        split_folder_dataset(guava, "Guava", samples)

    papaya = raw / "papaya main dataset"
    if papaya.is_dir():
        label_folder_dataset(papaya, "Papaya", samples, args.seed)

    wheat = raw / "wheat_leaf"
    if wheat.is_dir():
        def wheat_label(name: str) -> str:
            if name.casefold() == "healthy":
                disease = "Healthy"
            else:
                disease = clean_name(name)
            return class_name("Wheat", disease)

        label_folder_dataset(wheat, "Wheat", samples, args.seed, wheat_label)

    non_plant = raw / "not_a_plant"
    if non_plant.is_dir():
        label_folder_dataset(
            non_plant,
            "Other",
            samples,
            args.seed,
            lambda _: "Other___Not_a_Plant",
        )

    rice_crops = collect_rice_yolo(raw, output, samples)

    mango = raw / "mango_dataset"
    if mango.is_dir():
        print(
            "Skipping mango_dataset: its numeric YOLO class IDs have no class-name "
            "metadata, so their disease labels cannot be identified safely."
        )

    counts = stage_samples(samples, raw, output, args.copy)
    if not counts:
        raise SystemExit(f"No supported labeled images found under {raw}")
    print(f"Created {rice_crops} Rice object crops.")
    print("Prepared image counts:")
    for (split, label), count in sorted(counts.items()):
        print(f"  {split:5} {label}: {count}")
    print(f"Dataset ready at {output}")


if __name__ == "__main__":
    main()
