import streamlit as st
import requests
from PIL import Image
from io import BytesIO


# Page Config


st.set_page_config(
    page_title="Satellite Image Segmentation",
    page_icon="🛰️",
    layout="wide"
)


# Title


st.title("🛰️ Satellite Image Segmentation")
st.write(
    "Upload a satellite image and generate a segmentation mask using U-Net + EfficientNet-B3."
)


# Upload Image


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


# Prediction


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.divider()

    col1, col2 = st.columns(2)

   
    # Original Image
  

    with col1:
        st.subheader("Original Image")
        st.image(
            image,
            use_container_width=True
        )

 
    # Predict Button


    if st.button("Generate Segmentation Mask"):

        with st.spinner("Running model prediction..."):

            try:

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    files=files
                )

                if response.status_code == 200:

                    mask = Image.open(
                        BytesIO(response.content)
                    )

                    with col2:
                        st.subheader("Predicted Mask")
                        st.image(
                            mask,
                            use_container_width=True
                        )

                    st.success("Prediction completed successfully!")

                else:
                    st.error(
                        f"FastAPI Error: {response.status_code}"
                    )

            except Exception as e:

                st.error(
                    f"Unable to connect to FastAPI server.\n\n{e}"
                )


# Footer


st.divider()

st.caption(
    "Model: U-Net + EfficientNet-B3 | "
    "Framework: PyTorch | "
    "Deployment: FastAPI + Streamlit"
)