"""Train the LeafAI image classifier from prepared train/validation folders."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import tensorflow as tf

IMG_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/organized")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="backend/plant_disease_model.keras")
    parser.add_argument("--labels", default="backend/class_names.json")
    parser.add_argument("--initial-model", default="backend/plant_disease_model.keras")
    parser.add_argument(
        "--resume-model",
        type=Path,
        help="Continue training from a saved classifier with the same class ordering.",
    )
    parser.add_argument(
        "--resume-labels",
        type=Path,
        help="Class-name file for --resume-model; defaults to --labels.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data)
    train_dir = data_dir / "train"
    validation_dir = data_dir / "val"
    if not train_dir.is_dir() or not validation_dir.is_dir():
        raise SystemExit(f"Expected prepared train/ and val/ directories under {data_dir}")
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs and --batch-size must be positive.")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=args.batch_size,
        shuffle=True,
        seed=42,
    )
    class_names = train_ds.class_names
    if len(class_names) < 2:
        raise SystemExit("At least two classes with training images are required.")

    validation_ds = tf.keras.utils.image_dataset_from_directory(
        validation_dir,
        class_names=class_names,
        image_size=IMG_SIZE,
        batch_size=args.batch_size,
        shuffle=False,
    )
    if validation_ds.cardinality().numpy() == 0:
        raise SystemExit("The validation split contains no images.")

    class_counts = [
        sum(1 for path in (train_dir / name).iterdir() if path.is_file())
        for name in class_names
    ]
    if any(count == 0 for count in class_counts):
        raise SystemExit("Every class must have at least one training image.")
    mean_count = sum(class_counts) / len(class_counts)
    class_weights = tf.constant(
        [min(3.0, (mean_count / count) ** 0.5) for count in class_counts],
        dtype=tf.float32,
    )

    def prepare_batch(images, labels):
        weights = tf.gather(class_weights, labels)
        return images / 255.0, labels, weights

    train_ds = train_ds.map(prepare_batch, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    validation_ds = validation_ds.map(
        lambda images, labels: (images / 255.0, labels),
        num_parallel_calls=AUTOTUNE,
    ).prefetch(AUTOTUNE)

    if args.resume_model:
        if not args.resume_model.is_file():
            raise SystemExit(f"Resume model not found: {args.resume_model}")
        resume_labels = args.resume_labels or Path(args.labels)
        if not resume_labels.is_file():
            raise SystemExit(f"Resume class names not found: {resume_labels}")
        saved_class_names = json.loads(resume_labels.read_text(encoding="utf-8"))
        if saved_class_names != class_names:
            raise SystemExit(
                "Resume model class names do not match the prepared dataset ordering."
            )
        model = tf.keras.models.load_model(args.resume_model, compile=False)
        if model.input_shape != (None, *IMG_SIZE, 3) or model.output_shape[-1] != len(class_names):
            raise SystemExit(
                "Resume model must accept 224x224 RGB images and output one score per class."
            )
        print(f"Resuming classifier training from {args.resume_model}")
    else:
        base = tf.keras.applications.MobileNetV3Small(
            input_shape=(*IMG_SIZE, 3),
            include_top=False,
            weights=None,
            include_preprocessing=False,
        )
        initial_model = Path(args.initial_model)
        if initial_model.is_file():
            previous_model = tf.keras.models.load_model(initial_model, compile=False)
            previous_base = previous_model.get_layer("MobileNetV3Small")
            base.set_weights(previous_base.get_weights())
            print(f"Initialized MobileNetV3 from {initial_model}")
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
        outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4 if args.resume_model else 1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    output_path = Path(args.output)
    labels_path = Path(args.labels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f"{output_path.stem}.training.keras")
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
    os.replace(temporary_output, output_path)

    temporary_labels = labels_path.with_name(f"{labels_path.name}.training")
    temporary_labels.write_text(
        json.dumps(class_names, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_labels, labels_path)
    print(f"Saved model to {output_path}")
    print(f"Saved {len(class_names)} class labels to {labels_path}")


if __name__ == "__main__":
    main()
