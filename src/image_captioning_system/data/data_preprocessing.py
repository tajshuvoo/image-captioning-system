import re
from collections import defaultdict


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_image_caption_mapping(captions):
    image_captions = defaultdict(list)

    for line in captions:
        image_id, caption = line.split(",", 1)
        caption = clean_text(caption)
        caption = f"start {caption} end"
        image_captions[image_id].append(caption)

    return image_captions


def get_max_caption_length(image_captions):
    return max(
        len(caption.split())
        for captions in image_captions.values()
        for caption in captions
    )
