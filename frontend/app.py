import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="AI Plant Health Scanner", page_icon="🌿", layout="centered")

st.title("🌿 AI Plant Disease & Health Scanner")
st.caption("Mobile-friendly plant diagnostic engine powered by OpenCV & TFLite")

api_url = st.sidebar.text_input("Backend API URL", "http://localhost:8000/api/v1/scan")

input_mode = st.radio("Select Input Method:", ["Upload Image File", "Take Photo"], horizontal=True)

uploaded_file = None
if input_mode == "Upload Image File":
    uploaded_file = st.file_uploader("Choose a plant image...", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("Capture plant image")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(bytes_data))

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Plant Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Diagnostic Engine")
        st.write("Click below to process image through OpenCV preprocessing & TFLite inference.")
        scan_btn = st.button("🔍 Run Diagnostic Scan", type="primary", use_container_width=True)

    if scan_btn:
        with st.spinner("Processing leaf scan..."):
            try:
                files = {"file": ("leaf.jpg", bytes_data, "image/jpeg")}
                response = requests.post(api_url, files=files, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    confidence = data.get("confidence", 0.0)
                    report = data.get("report", {})

                    st.markdown("---")
                    st.subheader("📊 Diagnostic Results")

                    if confidence < 60.0:
                        st.warning(f"⚠️ Low confidence classification ({confidence:.1f}%). Please re-take a clearer picture.")
                    else:
                        st.success(f"**Diagnosis:** {report.get('disease_name')}")
                        st.progress(int(confidence) / 100)
                        st.caption(f"Model Confidence Score: **{confidence:.2f}%**")

                    tab1, tab2, tab3 = st.tabs(["📋 Symptoms", "🌱 Organic Cures", "🧪 Chemical Cures"])
                    with tab1:
                        st.info(report.get("symptoms", "N/A"))
                    with tab2:
                        st.success(report.get("organic_control", "N/A"))
                    with tab3:
                        st.warning(report.get("chemical_control", "N/A"))

                    st.caption(f"**Citation:** {report.get('source')}")

                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot reach FastAPI backend. Verify Uvicorn is running on port 8000.")
