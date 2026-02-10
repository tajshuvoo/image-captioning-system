import tensorflow as tf



import streamlit as st
import numpy as np
import pickle
from PIL import Image


# ----------------------------
# PATHS
# ----------------------------
MODEL_PATH = "streamlit_app/caption_model.keras"
TOKENIZER_PATH = "streamlit_app/tokenizer.pkl"
MAX_LEN_PATH = "streamlit_app/max_length.txt"

# ----------------------------
# LOAD MODELS (CACHED)
# ----------------------------
@st.cache_resource
def load_vgg():
    base = tf.keras.applications.VGG16(weights="imagenet")
    model = tf.keras.Model(
        inputs=base.inputs,
        outputs=base.layers[-2].output
    )
    return model

@st.cache_resource
def load_caption_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_max_length():
    with open(MAX_LEN_PATH) as f:
        return int(f.read())

vgg = load_vgg()
caption_model = load_caption_model()
tokenizer = load_tokenizer()
max_length = load_max_length()

# ----------------------------
# FEATURE EXTRACTION
# ----------------------------
def extract_features(image: Image.Image):
    image = image.resize((224, 224))
    img = tf.keras.preprocessing.image.img_to_array(image)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    feature = vgg.predict(img, verbose=0)
    return feature

# ----------------------------
# CAPTION GENERATION
# ----------------------------
def idx_to_word(idx, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == idx:
            return word
    return None

def predict_caption(model, feature, tokenizer, max_length):
    in_text = "start"

    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = tf.keras.preprocessing.sequence.pad_sequences(
            [seq], maxlen=max_length
        )

        yhat = model.predict([feature, seq], verbose=0)
        yhat = np.argmax(yhat)

        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break

        in_text += " " + word
        if word == "end":
            break

    # 🚨 remove start/end tokens
    words = in_text.split()
    words = [w for w in words if w not in ["start", "end"]]
    return " ".join(words)

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(
    page_title="Image Caption Generator",
    layout="centered"
)

st.title("🖼️ Image Caption Generator")
st.write("Upload an image and generate a caption using your trained model.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_column_width=True
    )

    with st.spinner("Generating caption..."):
        feature = extract_features(image)
        caption = predict_caption(
            caption_model,
            feature,
            tokenizer,
            max_length
        )

    st.success("Caption generated!")
    st.markdown(f"### 📝 {caption}")
