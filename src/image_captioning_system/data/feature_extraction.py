import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def load_inception_model():
    base_model = tf.keras.applications.inception_v3.InceptionV3(weights="imagenet")
    model = tf.keras.models.Model(
        inputs=base_model.inputs,
        outputs=base_model.layers[-2].output  # 2048-D
    )
    return model


def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(299, 299))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img


def extract_features(image_ids, images_dir, model):
    features = {}

    for image_id in tqdm(image_ids, desc="Extracting features"):
        image_path = os.path.join(images_dir, image_id)
        img = preprocess_image(image_path)
        feature = model.predict(img, verbose=0)
        features[image_id] = feature.flatten()

    return features
