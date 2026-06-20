import streamlit as st
import torch
import numpy as np
import cv2

from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as smp


# Page Config


st.set_page_config(
    page_title="Satellite Image Segmentation",
    page_icon="🛰️",
    layout="wide"
)


# Device


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Load Model (cached)


@st.cache_resource
def load_model():

    model = smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights=None,
        in_channels=3,
        classes=6
    )

    model.load_state_dict(
        torch.load(
            "best_model.pth",
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model


model = load_model()


# Transform


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Class Colors


CLASS_COLORS = {
    0: [60, 16, 152],      # Building
    1: [132, 41, 246],     # Land
    2: [110, 193, 228],    # Road
    3: [254, 221, 58],     # Vegetation
    4: [226, 169, 41],     # Water
    5: [155, 155, 155]     # Unlabeled
}


# Title


st.title("🛰️ Satellite Image Segmentation")

st.write(
    "Upload a satellite image and generate a segmentation mask using U-Net + EfficientNet-B3."
)


# Upload


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


# Prediction


if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    if st.button("Generate Segmentation Mask"):

        with st.spinner("Running model prediction..."):

            image_resized = image.resize((256, 256))

            image_np = np.array(image_resized)

            tensor = (
                transform(image_np)
                .unsqueeze(0)
                .to(device)
            )

            with torch.no_grad():

                output = model(tensor)

                pred = torch.argmax(
                    output,
                    dim=1
                ).squeeze().cpu().numpy()

            mask_rgb = np.zeros(
                (256, 256, 3),
                dtype=np.uint8
            )

            for cls, color in CLASS_COLORS.items():
                mask_rgb[pred == cls] = color

            mask_image = Image.fromarray(mask_rgb)

            with col2:
                st.subheader("Predicted Mask")
                st.image(
                    mask_image,
                    use_container_width=True
                )

            st.success("Prediction completed successfully!")


# Footer


st.divider()

st.caption(
    "Model: U-Net + EfficientNet-B3 | "
    "Framework: PyTorch | "
    "Deployment: Streamlit"
)