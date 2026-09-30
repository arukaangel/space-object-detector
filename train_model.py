from pathlib import Path
import json
import tensorflow as tf

DATASET_DIR = Path("dataset")
EXPORT_DIR = Path("exported_model")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS_HEAD = 8
EPOCHS_FINE = 4

CLASS_NAMES = ["planet", "star", "nebulae", "galaxy"]


def main():
    missing = [name for name in CLASS_NAMES if not (DATASET_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dataset folders: " + ", ".join(missing) +
            ". Create dataset/{planet,star,nebulae,galaxy} and add images."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        class_names=CLASS_NAMES,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        class_names=CLASS_NAMES,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
    ])

    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )

    base.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(
        len(CLASS_NAMES),
        activation="softmax"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("Training classifier head...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD
    )

    base.trainable = True

    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("Fine-tuning top MobileNetV2 layers...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE
    )

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    keras_path = EXPORT_DIR / "space_object_detector.h5"
    model.save(keras_path)

    with open(EXPORT_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(CLASS_NAMES, f, indent=2)

    print(f"Saved Keras model to {keras_path}")
    print("Next step: convert the model to TensorFlow.js with:")
    print("tensorflowjs_converter --input_format=keras exported_model/space_object_detector.h5 model/")


if __name__ == "__main__":
    main()