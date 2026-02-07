import yaml
from image_captioning_system.data.data_ingestion import load_captions
from image_captioning_system.data.data_preprocessing import (
    build_image_caption_mapping,
    get_max_caption_length
)


def main():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    captions = load_captions(config["captions_path"])
    image_captions = build_image_caption_mapping(captions)

    max_len = get_max_caption_length(image_captions)

    print(f"Total images with captions: {len(image_captions)}")
    print(f"Max caption length: {max_len}")


if __name__ == "__main__":
    main()
