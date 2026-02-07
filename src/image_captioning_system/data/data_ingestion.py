import os
from sklearn.model_selection import train_test_split


def validate_dataset(images_dir, captions_path):
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if not os.path.exists(captions_path):
        raise FileNotFoundError(f"Captions file not found: {captions_path}")

    images = os.listdir(images_dir)
    if len(images) == 0:
        raise ValueError("Images directory is empty")

    print(f"Found {len(images)} images and captions file.")


def load_captions(captions_path):
    with open(captions_path, "r") as f:
        captions = f.readlines()

    captions = [c.lower() for c in captions[1:]]
    return captions


def split_image_ids(images_dir, test_size, val_size, random_state):
    image_ids = os.listdir(images_dir)

    train_ids, temp_ids = train_test_split(
        image_ids, test_size=test_size, random_state=random_state
    )

    val_ids, test_ids = train_test_split(
        temp_ids, test_size=val_size, random_state=random_state
    )

    return train_ids, val_ids, test_ids
