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
    # Force CPU (NO cuDNN, NO Metal GPU)
    # -------------------------------------------------
    tf.config.set_visible_devices([], "GPU")

    # -------------------------------------------------
    # Load config
    # -------------------------------------------------
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    # -------------------------------------------------
    # Load tokenizer
    # -------------------------------------------------
    with open("artifacts/tokenizer/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    vocab_size = len(tokenizer.word_index) + 1

    # -------------------------------------------------
    # Load captions
    # -------------------------------------------------
    captions = load_captions(config["captions_path"])
    image_captions = build_image_caption_mapping(captions)
    max_len = get_max_caption_length(image_captions)

    # -------------------------------------------------
    # Load validation features
    # -------------------------------------------------
    val_features = np.load(
        "artifacts/features/val_features.npy",
        allow_pickle=True
    ).item()

    val_caps = {k: image_captions[k] for k in val_features.keys()}

    # -------------------------------------------------
    # 🔥 Rebuild model CLEANLY (NO cuDNN)
    # -------------------------------------------------
    model = build_caption_model(
        vocab_size=vocab_size,
        max_caption_length=max_len
    )

    model.load_weights("artifacts/model/inceptionmodel.keras")

    # -------------------------------------------------
    # Limit evaluation size (FAST)
    # -------------------------------------------------
    MAX_EVAL_IMAGES = 100
    image_ids = list(val_features.keys())[:MAX_EVAL_IMAGES]

    greedy_preds, beam_preds, references = [], [], []

    print(f"🚀 Evaluating on {len(image_ids)} validation images...")

    for image_id in image_ids:
        feature = val_features[image_id]

        refs = [
            cap.replace("start", "").replace("end", "").strip().split()
            for cap in val_caps[image_id]
        ]
        references.append(refs)

        greedy = generate_caption_greedy(
            model, tokenizer, feature, max_len
        )
        greedy_preds.append(greedy.split())

        beam = generate_caption_beam_search(
            model, tokenizer, feature, max_len, beam_width=3
        )
        beam_preds.append(beam.split())

    # -------------------------------------------------
    # BLEU Scores
    # -------------------------------------------------
    greedy_bleu = compute_bleu_scores(references, greedy_preds)
    beam_bleu = compute_bleu_scores(references, beam_preds)

    print("\n📊 BLEU SCORES (Validation)")
    for k, v in greedy_bleu.items():
        print(f"Greedy {k}: {v:.4f}")
    for k, v in beam_bleu.items():
        print(f"Beam   {k}: {v:.4f}")

    # -------------------------------------------------
    # MLflow
    # -------------------------------------------------
    with mlflow.start_run(run_name="stage_05_evaluation"):
        for k, v in greedy_bleu.items():
            mlflow.log_metric(f"greedy_{k}", v)
        for k, v in beam_bleu.items():
            mlflow.log_metric(f"beam_{k}", v)

    print("\n✅ Stage-05 evaluation complete.")


if __name__ == "__main__":
    main()
