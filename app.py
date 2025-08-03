import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load models
model_eff = load_model("models/tree_species_model.keras")
model_bn_old = load_model("models/best_cnn_model_plain.keras")
model_bn_best = load_model("models/best_cnn_model_batchnorm.keras")
model_bn_alt = load_model("models/final_cnn_batchnorm.keras")  # Use compile=False to reduce memory
model_plain = load_model("models/final_cnn_plain.keras")

# Tree species class names (in correct order)
class_names = [
    'amla', 'asopalav', 'babul', 'bamboo', 'banyan', 'bili', 'cactus', 'champa',
    'coconut', 'garmalo', 'gulmohor', 'gunda', 'jamun', 'kanchan', 'kesudo',
    'khajur', 'mango', 'motichanoti', 'neem', 'nilgiri', 'other', 'pilikaren',
    'pipal', 'saptaparni', 'shirish', 'simlo', 'sitafal', 'sonmahor',
    'sugarcane', 'vad'
]

# Preprocessing function
def preprocess_image(img_file):
    img = Image.open(img_file).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# Streamlit UI
st.title("🌳 Tree Species Classifier")
st.write("Upload a tree image to classify it using 5 different models.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    img_batch = preprocess_image(uploaded_file)

    # Predictions
    pred_eff = model_eff.predict(img_batch)
    pred_bn_old = model_bn_old.predict(img_batch)
    pred_bn_best = model_bn_best.predict(img_batch)
    pred_bn_alt = model_bn_alt.predict(img_batch)
    pred_plain = model_plain.predict(img_batch)

    class_eff = class_names[np.argmax(pred_eff)]
    class_bn_old = class_names[np.argmax(pred_bn_old)]
    class_bn_best = class_names[np.argmax(pred_bn_best)]
    class_bn_alt = class_names[np.argmax(pred_bn_alt)]
    class_plain = class_names[np.argmax(pred_plain)]

    # Optional: Guess from filename
    suggested_label = uploaded_file.name.split("_")[0].lower()

    # Display results
    st.subheader("Predicted Tree Species")
    st.write(f"🌿 Predicte image: **{class_eff}**")
    st.write(f"🌿 Best CNN with BatchNorm: **{class_bn_best}**")
    st.write(f"🌿 Alternate CNN with BatchNorm: **{class_bn_alt}**")
    st.write(f"🌿 Old CNN (Plain): **{class_bn_old}**")
    st.write(f"🌿 Plain CNN: **{class_plain}**")
    
    st.markdown("---")
    st.write(f"📂 File name suggests: **{suggested_label}**")
