
from pathlib import Path
import shutil
import yaml

ROOT = Path(r"D:\skin_disease")
OUTPUT = ROOT / "indexed_7class_clean"

# Original ID -> new ID
CLASS_MAP = {
    0: 0,    # Acne
    1: 1,    # Blackhead
    14: 2,   # Whitehead
    4: 3,    # Cystic
    9: 4,    # Papular
    10: 5,   # Purulent
    11: 6,   # Scars
}

CLASS_NAMES = [
    "Acne",
    "Blackhead",
    "Whitehead",
    "Cystic",
    "Papular",
    "Purulent",
    "Scars",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    # Verify source folders before processing
    for split in ("train", "valid", "test"):
        for kind in ("images", "labels"):
            folder = ROOT / split / kind
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing folder: {folder}")

    # Recreate only the generated output folder
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    total_images = 0
    total_boxes = 0

    for split in ("train", "valid", "test"):
        image_dir = ROOT / split / "images"
        label_dir = ROOT / split / "labels"
        out_images = OUTPUT / split / "images"
        out_labels = OUTPUT / split / "labels"

        out_images.mkdir(parents=True)
        out_labels.mkdir(parents=True)

        image_lookup = {
            p.stem: p
            for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        }

        indexed = 0
        boxes = 0
        skipped = 0
        missing = 0

        for label_file in sorted(label_dir.glob("*.txt")):
            print(f"Processing {split}: {label_file.name}", flush=True)
            image_file = image_lookup.get(label_file.stem)

            if image_file is None:
                missing += 1
                continue

            kept = []

            for line_no, line in enumerate(
                label_file.read_text(encoding="utf-8-sig").splitlines(),
                start=1,
            ):
                parts = line.split()
                if not parts:
                    continue

                try:
                    old_id = int(parts[0])
                except ValueError:
                    raise ValueError(
                        f"Invalid class ID: {label_file}, line {line_no}"
                    )

                if old_id not in CLASS_MAP:
                    continue

                if len(parts) != 5:
                    raise ValueError(
                        f"Invalid YOLO detection label: "
                        f"{label_file}, line {line_no}"
                    )

                # Remap ID; preserve the bounding-box coordinates
                parts[0] = str(CLASS_MAP[old_id])
                kept.append(" ".join(parts))

            # Skip images with none of the selected classes
            if not kept:
                skipped += 1
                continue

            shutil.copy2(image_file, out_images / image_file.name)
            (out_labels / label_file.name).write_text(
                "\n".join(kept) + "\n",
                encoding="utf-8",
            )

            indexed += 1
            boxes += len(kept)

        total_images += indexed
        total_boxes += boxes

        print(f"\n--- {split.upper()} ---")
        print(f"Indexed images: {indexed}")
        print(f"Retained annotations: {boxes}")
        print(f"Skipped images: {skipped}")
        print(f"Labels without matching images: {missing}")

    # Create the YOLO configuration
    data = {
        "path": OUTPUT.as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 7,
        "names": CLASS_NAMES,
    }

    with (OUTPUT / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print("\n========== INDEXING COMPLETE ==========")
    print(f"Total indexed images: {total_images}")
    print(f"Total retained annotations: {total_boxes}")
    print(f"Output: {OUTPUT}")
    print("Original dataset was not modified.")
    print("No training was performed.")


if __name__ == "__main__":
    main()
