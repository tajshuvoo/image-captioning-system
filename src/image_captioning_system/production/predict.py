import os
import pickle
import numpy as np
import tensorflow as tf

from image_captioning_system.data.feature_extraction import (
    load_feature_extractor,
    extract_single_image_feature,
)
from image_captioning_system.inference.greedy import generate_caption_greedy
from image_captioning_system.inference.beam_search import generate_caption_beam_search


# --------------------------------------------------
# CONFIG (hard-coded for production = safe)
# --------------------------------------------------
MODEL_PATH = "artifacts/model/inceptionmodel.keras"
TOKENIZER_PATH = "artifacts/tokenizer/tokenizer.pkl"

MAX_CAPTION_LENGTH = 37          # must match training
CNN_FEATURE_DIM = 2048


# --------------------------------------------------
# LOAD EVERYTHING ONCE
# --------------------------------------------------
def load_production_assets():
    print("🔄 Loading production assets...")

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    feature_extractor = load_feature_extractor()

    print("✅ Assets loaded")
    return model, tokenizer, feature_extractor


# --------------------------------------------------
# MAIN PREDICTION FUNCTION
# --------------------------------------------------
def predict_caption(
    image_path: str,
    mode: str = "beam",
    beam_width: int = 3,
):
    """
    mode: 'greedy' | 'beam'
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Extract CNN features
    image_feature = extract_single_image_feature(
        image_path,
        feature_extractor
    )

    image_feature = image_feature.reshape((CNN_FEATURE_DIM,))

    # Generate caption
    if mode == "greedy":
        caption = generate_caption_greedy(
            model,
            tokenizer,
            image_feature,
            MAX_CAPTION_LENGTH
        )
    else:
        caption = generate_caption_beam_search(
            model,
            tokenizer,
            image_feature,
            MAX_CAPTION_LENGTH,
            beam_width=beam_width
        )

    return caption


# --------------------------------------------------
# CLI ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--mode", default="beam", choices=["greedy", "beam"])
    parser.add_argument("--beam_width", type=int, default=3)

    args = parser.parse_args()

    model, tokenizer, feature_extractor = load_production_assets()

    caption = predict_caption(
        image_path=args.image,
        mode=args.mode,
        beam_width=args.beam_width,
    )

    print("\n🖼️ IMAGE:", args.image)
    print("📝 CAPTION:", caption)
