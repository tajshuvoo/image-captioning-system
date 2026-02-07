import yaml
from image_captioning_system.data.data_ingestion import (
    validate_dataset,
    load_captions,
    split_image_ids
)


def main():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    validate_dataset(
        config["images_dir"],
        config["captions_path"]
    )

    captions = load_captions(config["captions_path"])

    train_ids, val_ids, test_ids = split_image_ids(
        config["images_dir"],
        config["split"]["test_size"],
        config["split"]["val_size"],
        config["split"]["random_state"]
    )

    print(f"Train images: {len(train_ids)}")
    print(f"Validation images: {len(val_ids)}")
    print(f"Test images: {len(test_ids)}")
    print(f"Total captions: {len(captions)}")


if __name__ == "__main__":
    main()
