import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load models
model_eff = load_model("models/tree_species_model.keras", compile=False)
model_bn_old = load_model("models/best_cnn_model_plain.keras", compile=False)
model_bn_best = load_model("models/best_cnn_model_batchnorm.keras", compile=False)
model_bn_alt = load_model("models/final_cnn_batchnorm.keras", compile=False)
model_plain = load_model("models/final_cnn_plain.keras", compile=False)

# Class names
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
st.title("🌳 Tree Species Classifier - Combined Model Prediction")
st.write("Upload a tree image and get predictions from all 5 models and a final combined result.")

uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    img_batch = preprocess_image(uploaded_file)

    # Get predictions from all models
    preds = [
        model_eff.predict(img_batch),
        model_bn_old.predict(img_batch),
        model_bn_best.predict(img_batch),
        model_bn_alt.predict(img_batch),
        model_plain.predict(img_batch)
    ]

    final_class = class_names[np.argmax(preds[0])]

    # Show individual model predictions
    st.subheader("📊 Individual Model Predictions:")
    model_names = ["Efficient Model", "Old Plain CNN", "Best BatchNorm CNN", "Alt BatchNorm CNN", "Final Plain CNN"]

    for i, pred in enumerate(preds):
        pred_class = class_names[np.argmax(pred)]
        st.write(f"🔹 {model_names[i]}: **{pred_class}**")

    
    st.markdown("---")
    st.subheader("✅ Final Combined Prediction")
    st.success(f"🌿 **Predicted Tree Species: {final_class}**")

    # Optional: Suggested label from filename
    suggested_label = uploaded_file.name.split("_")[0].lower()
    st.write(f"📂 File name suggests: **{suggested_label}**")
