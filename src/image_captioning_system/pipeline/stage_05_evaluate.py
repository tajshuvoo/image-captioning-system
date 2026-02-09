import os
import pickle
import yaml
import mlflow
import numpy as np
import tensorflow as tf

from image_captioning_system.data.data_ingestion import load_captions
from image_captioning_system.data.data_preprocessing import (
    build_image_caption_mapping,
    get_max_caption_length,
)
from image_captioning_system.model.model import build_caption_model
from image_captioning_system.inference.greedy import generate_caption_greedy
from image_captioning_system.inference.beam_search import generate_caption_beam_search
from image_captioning_system.utils.metrics import compute_bleu_scores


def main():
    # -------------------------------------------------
    # Load config
    # -------------------------------------------------
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    # -------------------------------------------------
    # Paths
    # -------------------------------------------------
    MODEL_PATH = "artifacts/model/inceptionmodel.keras"
    TOKENIZER_PATH = "artifacts/tokenizer/tokenizer.pkl"

    # -------------------------------------------------
    # Load tokenizer
    # -------------------------------------------------
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    vocab_size = len(tokenizer.word_index) + 1

    # -------------------------------------------------
    # Load validation features
    # -------------------------------------------------
    val_features = np.load(
        "artifacts/features/val_features.npy",
        allow_pickle=True
    ).item()

    # -------------------------------------------------
    # Load captions → VAL ONLY
    # -------------------------------------------------
    captions = load_captions(config["captions_path"])
    image_captions = build_image_caption_mapping(captions)

    val_caps = {
        k: image_captions[k]
        for k in val_features.keys()
    }

    max_len = get_max_caption_length(image_captions)

    # -------------------------------------------------
    # 🔥 OPTION A FIX
    # Rebuild model WITHOUT cuDNN, then load weights
    # -------------------------------------------------
    model = build_caption_model(
        vocab_size=vocab_size,
        max_caption_length=max_len,
        cnn_output_dim=2048
    )

    model.load_weights(MODEL_PATH)

    # -------------------------------------------------
    # Generate predictions
    # -------------------------------------------------
    greedy_preds = []
    beam_preds = []
    references = []

    print("🚀 Generating captions on validation set...")

    for image_id, feature in val_features.items():
        # Ground-truth references
        refs = [
            cap.replace("start", "").replace("end", "").strip().split()
            for cap in val_caps[image_id]
        ]
        references.append(refs)

        # Greedy decoding
        greedy_caption = generate_caption_greedy(
            model,
            tokenizer,
            feature,
            max_len
        )
        greedy_preds.append(greedy_caption.split())

        # Beam search decoding
        beam_caption = generate_caption_beam_search(
            model,
            tokenizer,
            feature,
            max_len,
            beam_width=3
        )
        beam_preds.append(beam_caption.split())

    # -------------------------------------------------
    # Compute BLEU-1 → BLEU-4
    # -------------------------------------------------
    greedy_bleu = compute_bleu_scores(references, greedy_preds)
    beam_bleu = compute_bleu_scores(references, beam_preds)

    print("\n📊 BLEU SCORES (Validation Set)")
    print("Greedy:")
    for k, v in greedy_bleu.items():
        print(f"  {k}: {v:.4f}")

    print("\nBeam Search:")
    for k, v in beam_bleu.items():
        print(f"  {k}: {v:.4f}")

    # -------------------------------------------------
    # MLflow logging
    # -------------------------------------------------
    with mlflow.start_run(run_name="stage_05_evaluation"):
        for k, v in greedy_bleu.items():
            mlflow.log_metric(f"greedy_{k}", v)

        for k, v in beam_bleu.items():
            mlflow.log_metric(f"beam_{k}", v)

        mlflow.log_artifact(MODEL_PATH)
        mlflow.log_artifact(TOKENIZER_PATH)

    print("\n✅ Stage-05 evaluation complete.")


if __name__ == "__main__":
    main()
