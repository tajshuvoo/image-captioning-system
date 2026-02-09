import numpy as np
import tensorflow as tf


def generate_caption_beam_search(
    model,
    tokenizer,
    image_feature,
    max_caption_length,
    beam_width=3
):
    """
    Generate caption using Beam Search
    """

    start_token = tokenizer.word_index["start"]
    end_token = tokenizer.word_index["end"]

    sequences = [[
        [start_token],
        0.0
    ]]

    for _ in range(max_caption_length):
        all_candidates = []

        for seq, score in sequences:
            if seq[-1] == end_token:
                all_candidates.append((seq, score))
                continue

            padded_seq = tf.keras.preprocessing.sequence.pad_sequences(
                [seq],
                maxlen=max_caption_length
            )

            preds = model.predict(
                [image_feature.reshape(1, -1), padded_seq],
                verbose=0
            )[0]

            top_indices = np.argsort(preds)[-beam_width:]

            for idx in top_indices:
                prob = preds[idx]
                candidate = [
                    seq + [idx],
                    score - np.log(prob + 1e-9)
                ]
                all_candidates.append(candidate)

        # Select best beams
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        sequences = ordered[:beam_width]

    best_sequence = sequences[0][0]

    # Convert indices to words
    caption_words = []
    for idx in best_sequence:
        word = tokenizer.index_word.get(idx)
        if word in ["start", None]:
            continue
        if word == "end":
            break
        caption_words.append(word)

    return " ".join(caption_words)
