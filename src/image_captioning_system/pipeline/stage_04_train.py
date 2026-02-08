import os
import yaml
import pickle
import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf

from image_captioning_system.data.data_ingestion import load_captions
from image_captioning_system.data.data_preprocessing import (
    build_image_caption_mapping,
    get_max_caption_length
)
from image_captioning_system.model.model import build_caption_model
from image_captioning_system.model.train import data_generator


def main():
    # -------------------------------------------------
    # TensorFlow safety (macOS M1)
    # -------------------------------------------------
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)

    # -------------------------------------------------
    # Load config
    # -------------------------------------------------
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    # -------------------------------------------------
    # Load features
    # -------------------------------------------------
    train_features = np.load(
        "artifacts/features/train_features.npy",
        allow_pickle=True
    ).item()

    val_features = np.load(
        "artifacts/features/val_features.npy",
        allow_pickle=True
    ).item()

    # -------------------------------------------------
    # Load captions
    # -------------------------------------------------
    captions = load_captions(config["captions_path"])
    image_captions = build_image_caption_mapping(captions)
    max_len = get_max_caption_length(image_captions)

    # -------------------------------------------------
    # Tokenizer
    # -------------------------------------------------
    os.makedirs("artifacts/tokenizer", exist_ok=True)
    tokenizer_path = "artifacts/tokenizer/tokenizer.pkl"

    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
        print("Loaded existing tokenizer.")
    else:
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(
            cap
            for caps in image_captions.values()
            for cap in caps
        )
        with open(tokenizer_path, "wb") as f:
            pickle.dump(tokenizer, f)
        print("Tokenizer created and saved.")

    vocab_size = len(tokenizer.word_index) + 1

    # -------------------------------------------------
    # Train / Val split
    # -------------------------------------------------
    train_caps = {k: image_captions[k] for k in train_features}
    val_caps = {k: image_captions[k] for k in val_features}

    # -------------------------------------------------
    # Build model
    # -------------------------------------------------
    model = build_caption_model(vocab_size, max_len)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-4,   # 🔥 stable
            clipnorm=1.0          # 🔥 prevents exploding gradients
        ),
        loss="sparse_categorical_crossentropy"  # 🔥 MUST match generator
    )

    # -------------------------------------------------
    # Generators (NO tf.data)
    # -------------------------------------------------
    batch_size = 16  # safe + fast on 16GB RAM

    train_gen = data_generator(
        train_caps,
        train_features,
        tokenizer,
        max_len,
        vocab_size,
        batch_size
    )

    val_gen = data_generator(
        val_caps,
        val_features,
        tokenizer,
        max_len,
        vocab_size,
        batch_size
    )

    steps_per_epoch = sum(len(v) for v in train_caps.values()) // batch_size
    val_steps = sum(len(v) for v in val_caps.values()) // batch_size

    # -------------------------------------------------
    # Callbacks (NO EARLY STOP)
    # -------------------------------------------------
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=1,
            min_lr=1e-5,
            verbose=1
        )
    ]

    # -------------------------------------------------
    # MLflow
    # -------------------------------------------------
    mlflow.tensorflow.autolog(
        log_models=True,
        registered_model_name="inceptionmodel"
    )

    # -------------------------------------------------
    # TRAIN (FULL TRAINING)
    # -------------------------------------------------
    with mlflow.start_run():
        model.fit(
            train_gen,
            epochs=15,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=val_steps,
            callbacks=callbacks
        )

        # -------------------------------------------------
        # Save model (KERAS FORMAT)
        # -------------------------------------------------
        os.makedirs("artifacts/model", exist_ok=True)
        model_path = "artifacts/model/inceptionmodel.keras"
        tf.keras.saving.save_model(model, model_path)

        mlflow.log_artifact(model_path)
        mlflow.log_artifact(tokenizer_path)

    print("✅ Training + MLflow logging complete.")


if __name__ == "__main__":
    main()
