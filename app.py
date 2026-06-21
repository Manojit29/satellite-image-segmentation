
import streamlit as st
import torch
import numpy as np

from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as smp

st.set_page_config(
    page_title="Satellite Image Segmentation",
    page_icon="🛰️",
    layout="wide"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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
            map_location=device,
            weights_only=True
        )
    )

    model.to(device)
    model.eval()

    return model

model = load_model()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

CLASS_COLORS = {
    0: [60, 16, 152],      # Building
    1: [132, 41, 246],     # Land
    2: [110, 193, 228],    # Road
    3: [254, 221, 58],     # Vegetation
    4: [226, 169, 41],     # Water
    5: [155, 155, 155]     # Unlabeled
}

CLASS_NAMES = {
    0: "Building",
    1: "Land",
    2: "Road",
    3: "Vegetation",
    4: "Water",
    5: "Unlabeled"
}

st.title("🛰️ Satellite Image Segmentation")

st.write(
    "Upload a satellite image and generate a segmentation mask using U-Net + EfficientNet-B3."
)

st.markdown("### 🗺️ Segmentation Class Legend")

legend_cols = st.columns(3)

for idx, (cls, color) in enumerate(CLASS_COLORS.items()):

    hex_color = "#{:02X}{:02X}{:02X}".format(
        color[0],
        color[1],
        color[2]
    )

    with legend_cols[idx % 3]:
        st.markdown(
            f"""
            <div style="
                display:flex;
                align-items:center;
                margin:10px 0;
            ">
                <div style="
                    width:25px;
                    height:25px;
                    background:{hex_color};
                    border:1px solid black;
                    margin-right:10px;
                "></div>
                <span>{CLASS_NAMES[cls]}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

st.divider()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(
            image,
            use_container_width=True
        )

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

            st.success(
                "Prediction completed successfully!"
            )

st.divider()

st.caption(
    "Model: U-Net + EfficientNet-B3 | Framework: PyTorch | Deployment: Streamlit"
)
