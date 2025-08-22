import streamlit as st
import numpy as np
import tensorflow as tf
import os
import uuid
from PIL import Image
from fpdf import FPDF
import base64
import cv2
import pandas as pd
import textwrap
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
import google.generativeai as genai

# ====== CONFIG ======
st.set_page_config(page_title="BASIRA - OCT Diagnosis", layout="centered")
# ====== LOGO HEADER IMAGES ======
left_col, center_col, right_col = st.columns([5, 20, 5])

with left_col:
    st.image("logo1.png", width=100)  # Replace with your left image path

with center_col:
    st.title(" BASIRA : Biomedical AI for Smart Image Retinal Analysis ")
    st.subheader(" OCT Eye Disease Diagnosis ")

with right_col:
    st.image("logo.png", width=100) 


class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
MODEL_PATH = "DenseNet121_best_weights.keras"
GEMINI_API_KEY = "AIzaSyBjWR_RpbW9T6Vp91slyEfM_XnOZ-YDFjM"

# ====== INIT GEMINI CLIENT ======
genai.configure(api_key=GEMINI_API_KEY)

# ====== LOAD CLASSIFIER ======
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

classifier = load_model()

# ====== IMAGE PREPROCESSING ======
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    image = image.convert("RGB")
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# ====== GET LAST CONV LAYER ======
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

# ====== GRAD-CAM ======
def compute_gradcam(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    img = tf.squeeze(img_array).numpy()
    img = (img - img.min()) / (img.max() - img.min())
    img = np.uint8(255 * img)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    superimposed_img = cv2.addWeighted(img, 0.7, cv2.applyColorMap(heatmap, cv2.COLORMAP_JET), 0.3, 0)
    return Image.fromarray(superimposed_img)


# ====== GEMINI EXPLAINER ======
def ask_gemini(diagnosis):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            f"""
You are a professional ophthalmologist and medical researcher specialized in retinal diseases. Based on the OCT image diagnosis result below:

Diagnosis: {diagnosis}  
Confidence: {confidence:.2f}%

Please write a detailed medical report in English that includes the following sections:

1. Case Description: Provide a brief overview of the diagnosed retinal condition.
2. Symptoms: List the common clinical symptoms observed in patients with this condition.
3. Causes: Explain the main underlying causes or risk factors that lead to this disease.
4. Treatment Options: Outline the available treatment or management approaches, including medications, procedures, or lifestyle recommendations.

Make sure the information is accurate, concise, and suitable for a medical report format.
"""
        )
        return response.text
    except Exception as e:
        return f"Gemini error: {e}"


# ====== PDF GENERATOR ======
def generate_pdf_report(diagnosis, confidence, probs, gradcam_img: Image.Image, gemini_text: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ===== Title =====
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "BASIRA OCT Diagnosis Report", ln=True, align="C")

    # ===== Diagnosis =====
    pdf.set_font("Arial", '', 12)
    pdf.ln(10)
    pdf.multi_cell(0, 10, f"Diagnosis: {diagnosis} ({confidence:.2f}%)")

    # ===== Probabilities =====
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Prediction Probabilities:", ln=True)
    pdf.set_font("Arial", '', 12)
    for cls, prob in zip(class_names, probs):
        pdf.cell(0, 10, f"- {cls}: {prob:.2f}%", ln=True)

    # ===== Grad-CAM Image =====
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Grad-CAM Visualization:", ln=True)
    gradcam_path = f"gradcam_temp_{uuid.uuid4().hex[:6]}.jpg"
    gradcam_img.save(gradcam_path)
    pdf.image(gradcam_path, x=30, w=150)

    try:
        os.remove(gradcam_path)
    except:
        pass

    # ===== Medical Insights =====
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Medical Insights:", ln=True)
    pdf.set_font("Arial", '', 11)



    # ===== Save PDF =====
    pdf_path = f"OCT_Report_{uuid.uuid4().hex[:8]}.pdf"
    pdf.output(pdf_path)
    return pdf_path

# ====== STREAMLIT APP ======
uploaded_file = st.file_uploader("Upload an OCT Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded OCT Image", use_column_width=True)

    with st.spinner("🧠 Diagnosing..."):
        img_array = preprocess_image(image)
        preds = classifier.predict(img_array)
        pred_index = np.argmax(preds[0])
        class_name = class_names[pred_index]
        confidence = preds[0][pred_index]

        st.subheader("🩺 Diagnosis Result:")
        st.success(f"**{class_name}** ({confidence*100:.2f}%)")

        st.subheader("📊 Prediction Probabilities:")
        for i, prob in enumerate(preds[0]):
            st.write(f"{class_names[i]}: {prob*100:.2f}%")

        # Optional: Probability chart
        st.bar_chart(pd.DataFrame({
            'Class': class_names,
            'Probability': [p * 100 for p in preds[0]]
        }).set_index("Class"))

        st.subheader("🔥 Grad-CAM:")
        grad_layer = get_last_conv_layer(classifier)
        gradcam_image = compute_gradcam(classifier, img_array, grad_layer)
        st.image(gradcam_image, caption="Grad-CAM Heatmap", use_column_width=True)

        st.subheader("💡 Medical Insight :")
        info = ask_gemini(class_name)
        st.markdown(info)

        if st.button("Generate Diagnosis Report (PDF)"):
            with st.spinner("Creating PDF report..."):
                pdf_path = generate_pdf_report(
                    class_name,
                    confidence * 100,
                    [p * 100 for p in preds[0]],
                    gradcam_image,
                    info
                )
                with open(pdf_path, "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                    pdf_display = f'<a href="data:application/pdf;base64,{base64_pdf}" download="OCT_Report.pdf">📥 Click to download your PDF Report</a>'
                    st.markdown(pdf_display, unsafe_allow_html=True)
