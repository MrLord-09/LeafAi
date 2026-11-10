"""Train a binary plant/non-plant detector from the Leafybud2 dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import tensorflow as tf

IMG_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE
SOURCE_CLASSES = {"not plant": 0, "plant": 1}
CLASS_NAMES = ["not_plant", "plant"]
SPLITS = {
    "train": "1 training",
    "validation": "2 validating",
    "test": "3 testing",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
ROOT = Path(__file__).resolve().parents[1]


def load_split(data_dir: Path, split_name: str, batch_size: int, shuffle: bool) -> tf.data.Dataset:
    split_dir = data_dir / SPLITS[split_name]
    if not split_dir.is_dir():
        raise SystemExit(f"Expected {split_name} split directory: {split_dir}")

    class_dirs = {
        path.name.casefold(): path
        for path in split_dir.iterdir()
        if path.is_dir()
    }
    if set(class_dirs) != set(SOURCE_CLASSES):
        expected = ", ".join(sorted(SOURCE_CLASSES))
        actual = ", ".join(sorted(class_dirs)) or "none"
        raise SystemExit(
            f"Expected only the '{expected}' class folders under {split_dir}; found: {actual}"
        )

    image_paths: list[str] = []
    labels: list[int] = []
    for class_name, label in SOURCE_CLASSES.items():
        files = sorted(
            path
            for path in class_dirs[class_name].rglob("*")
            if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
        )
        if not files:
            raise SystemExit(f"No supported images found in {class_dirs[class_name]}")
        image_paths.extend(str(path) for path in files)
        labels.extend([label] * len(files))

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def decode_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.decode_image(
            tf.io.read_file(path),
            channels=3,
            expand_animations=False,
        )
        image.set_shape((None, None, 3))
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    if shuffle:
        dataset = dataset.shuffle(min(len(image_paths), 4096), seed=42, reshuffle_each_iteration=True)
    return (
        dataset
        .map(decode_image, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "raw" / "Leafybud2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=ROOT / "backend" / "plant_detector.keras")
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "backend" / "plant_detector_class_names.json",
    )
    parser.add_argument(
        "--initial-model",
        type=Path,
        default=ROOT / "backend" / "plant_disease_model.keras",
        help="Keras model used to initialize the MobileNetV3 feature extractor.",
    )
    parser.add_argument(
        "--resume-model",
        type=Path,
        help="Continue training from a saved two-class plant detector checkpoint.",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs and --batch-size must be positive.")

    train_ds = load_split(args.data, "train", args.batch_size, shuffle=True)
    validation_ds = load_split(args.data, "validation", args.batch_size, shuffle=False)
    test_ds = load_split(args.data, "test", args.batch_size, shuffle=False)

    if args.resume_model:
        if not args.resume_model.is_file():
            raise SystemExit(f"Resume model not found: {args.resume_model}")
        model = tf.keras.models.load_model(args.resume_model, compile=False)
        if model.input_shape != (None, *IMG_SIZE, 3) or model.output_shape[-1] != len(CLASS_NAMES):
            raise SystemExit("Resume model must accept 224x224 RGB images and output two classes.")
        print(f"Resuming plant/non-plant training from {args.resume_model}")
    else:
        base = tf.keras.applications.MobileNetV3Small(
            input_shape=(*IMG_SIZE, 3),
            include_top=False,
            weights=None,
            include_preprocessing=False,
        )
        if args.initial_model.is_file():
            previous_model = tf.keras.models.load_model(args.initial_model, compile=False)
            base.set_weights(previous_model.get_layer("MobileNetV3Small").get_weights())
            print(f"Initialized MobileNetV3 from {args.initial_model}")
        else:
            imagenet_model = tf.keras.applications.MobileNetV3Small(
                input_shape=(*IMG_SIZE, 3),
                include_top=False,
                weights="imagenet",
                include_preprocessing=False,
            )
            base.set_weights(imagenet_model.get_weights())
            print("Initialized MobileNetV3 with ImageNet weights.")
        base.trainable = False

        augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.08),
                tf.keras.layers.RandomZoom(0.15),
                tf.keras.layers.RandomContrast(0.1),
            ],
            name="augmentation",
        )
        inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
        x = augmentation(inputs)
        x = tf.keras.layers.Rescaling(2.0, offset=-1.0)(x)
        x = base(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.labels.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(f"{args.output.stem}.checkpoint.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(temporary_output, save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.3),
    ]
    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    model.save(temporary_output)
    os.replace(temporary_output, args.output)

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=2)
    temporary_labels = args.labels.with_name(f"{args.labels.name}.training")
    temporary_labels.write_text(
        json.dumps(CLASS_NAMES, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_labels, args.labels)
    print(f"Test loss: {test_loss:.4f}; test accuracy: {test_accuracy:.4%}")
    print(f"Saved model to {args.output}")
    print(f"Saved {len(CLASS_NAMES)} class labels to {args.labels}")


if __name__ == "__main__":
    main()