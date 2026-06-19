from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

import torch
import numpy as np
import cv2

from PIL import Image
from torchvision import transforms

import segmentation_models_pytorch as smp


# FastAPI App


app = FastAPI(
    title="Satellite Image Segmentation API",
    description="U-Net + EfficientNet-B3 Satellite Segmentation",
    version="1.0"
)


# Device


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Model


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

print("Model loaded successfully")
print("Using device:", device)
# Transform


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Dataset Class Colors


CLASS_COLORS = {
    0: [60, 16, 152],      # Building
    1: [132, 41, 246],     # Land
    2: [110, 193, 228],    # Road
    3: [254, 221, 58],     # Vegetation
    4: [226, 169, 41],     # Water
    5: [155, 155, 155]     # Unlabeled
}



# Home Route


@app.get("/")
def home():
    return {
        "message": "Satellite Image Segmentation API Running"
    }

# Prediction Route


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read image
    image = Image.open(file.file).convert("RGB")

    # Resize to model input size
    image = image.resize((256, 256))

    image_np = np.array(image)

    # Preprocess
    tensor = transform(image_np).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # Create RGB mask
    mask_rgb = np.zeros((256, 256, 3), dtype=np.uint8)

    for cls, color in CLASS_COLORS.items():
        mask_rgb[pred == cls] = color

    # Save mask
    output_path = "prediction.png"

    cv2.imwrite(
        output_path,
        cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    )

    return FileResponse(
        output_path,
        media_type="image/png",
        filename="prediction.png"
    )