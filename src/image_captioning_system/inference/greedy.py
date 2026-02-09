import numpy as np
import tensorflow as tf


def generate_caption_greedy(
    model,
    tokenizer,
    image_feature,
    max_caption_length,
):
    """
    Generate caption using Greedy Search
    """

    in_text = "start"

    for _ in range(max_caption_length):
        # Convert text to sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = tf.keras.preprocessing.sequence.pad_sequences(
            [sequence],
            maxlen=max_caption_length
        )

        # Predict next word
        yhat = model.predict(
            [image_feature.reshape(1, -1), sequence],
            verbose=0
        )

        yhat_index = np.argmax(yhat)

        # Convert index to word
        word = tokenizer.index_word.get(yhat_index)

        if word is None:
            break

        in_text += " " + word

        if word == "end":
            break

    # Cleanup tokens
    caption = in_text.replace("start", "").replace("end", "").strip()
    return caption
