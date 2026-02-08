import tensorflow as tf


def build_caption_model(vocab_size, max_caption_length, cnn_output_dim=2048):
    # -------------------------------
    # Image feature branch
    # -------------------------------
    image_input = tf.keras.layers.Input(
        shape=(cnn_output_dim,),
        name="image_features"
    )
    fe1 = tf.keras.layers.BatchNormalization()(image_input)
    fe2 = tf.keras.layers.Dense(256, activation="relu")(fe1)
    fe3 = tf.keras.layers.BatchNormalization()(fe2)

    # -------------------------------
    # Caption branch
    # -------------------------------
    caption_input = tf.keras.layers.Input(
        shape=(max_caption_length,),
        name="caption_input"
    )

    se1 = tf.keras.layers.Embedding(
        vocab_size,
        256,
        mask_zero=True
    )(caption_input)

    # 🔥 FIX: disable cuDNN
    se2 = tf.keras.layers.LSTM(
        256
    )(se1)

    # -------------------------------
    # Decoder
    # -------------------------------
    decoder1 = tf.keras.layers.add([fe3, se2])
    decoder2 = tf.keras.layers.Dense(256, activation="relu")(decoder1)
    output = tf.keras.layers.Dense(
        vocab_size,
        activation="softmax"
    )(decoder2)

    model = tf.keras.models.Model(
        inputs=[image_input, caption_input],
        outputs=output,
        name="image_caption_model"
    )

    return model
