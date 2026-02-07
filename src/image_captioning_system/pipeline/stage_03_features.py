import os
import yaml
import numpy as np
from image_captioning_system.data.feature_extraction import (
    load_inception_model,
    extract_features
)
from image_captioning_system.data.data_ingestion import split_image_ids


def main():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    images_dir = config["images_dir"]

    train_ids, val_ids, test_ids = split_image_ids(
        images_dir,
        config["split"]["test_size"],
        config["split"]["val_size"],
        config["split"]["random_state"]
    )

    model = load_inception_model()

    os.makedirs("artifacts/features", exist_ok=True)

    train_features = extract_features(train_ids, images_dir, model)
    val_features = extract_features(val_ids, images_dir, model)
    test_features = extract_features(test_ids, images_dir, model)

    np.save("artifacts/features/train_features.npy", train_features)
    np.save("artifacts/features/val_features.npy", val_features)
    np.save("artifacts/features/test_features.npy", test_features)

    print("Feature extraction completed and saved.")


if __name__ == "__main__":
    main()
