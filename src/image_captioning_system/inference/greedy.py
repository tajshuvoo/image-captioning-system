import numpy as np
import tensorflow as tf


def generate_caption_greedy(model, tokenizer, image_feature, max_len):
    in_text = "start"
    used_words = set()

    for _ in range(max_len):
        sequence = tokenizer.texts_to_sequences([in_text])[0]

        sequence = tf.keras.preprocessing.sequence.pad_sequences(
            [sequence],
            maxlen=max_len,
            padding="post"
        )

        yhat = model.predict(
            [image_feature.reshape(1, -1), sequence],
            verbose=0
        )[0]

        yhat_idx = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_idx)

        if word is None:
            break

        # 🔥 REPETITION STOP
        if word in used_words:
            break
        used_words.add(word)

        in_text += " " + word

        if word == "end":
            break

    return in_text.replace("start", "").replace("end", "").strip()
