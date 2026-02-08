import numpy as np
import tensorflow as tf


def data_generator(
    captions_map,
    image_features,
    tokenizer,
    max_caption_length,
    vocab_size,
    batch_size
):
    image_ids = list(captions_map.keys())

    while True:
        X_img, X_seq, y = [], [], []

        for image_id in image_ids:
            feature = image_features[image_id]
            captions = captions_map[image_id]

            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]

                for i in range(1, len(seq)):
                    in_seq = seq[:i]
                    out_seq = seq[i]  # 🔥 integer label (FAST)

                    in_seq = tf.keras.preprocessing.sequence.pad_sequences(
                        [in_seq],
                        maxlen=max_caption_length
                    )[0]

                    X_img.append(feature)
                    X_seq.append(in_seq)
                    y.append(out_seq)

                    if len(X_img) == batch_size:
                        yield (
                            (
                                np.array(X_img, dtype="float32"),
                                np.array(X_seq, dtype="int32")
                            ),
                            np.array(y, dtype="int32")  # 🔥 sparse labels
                        )

                        X_img, X_seq, y = [], [], []
