import base64
import cv2
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from torchvision import models, transforms
from PIL import Image
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware

# ================= CONFIG =================
MODEL_PATH = "age_model.pth"
IMG_SIZE = 224
STABLE_FRAMES = 3
# =========================================

app = FastAPI(title="Age Safe Reels API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for testing only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Device --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Load Model --------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# -------- Transform --------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------- Face Detector --------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------- Age Category --------
def age_category(age):
    age = int(round(age))
    if age <= 12:
        return "Kid"
    elif age <= 17:
        return "Teen"
    elif age <= 25:
        return "Young Adult"
    elif age <= 59:
        return "Adult"
    else:
        return "Senior"


# -------- Request Model --------
class ImagePayload(BaseModel):
    image: str  # base64 image


@app.post("/api/age-check")
def age_check(payload: ImagePayload):
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(payload.image)
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        frame = np.array(img)
        # Resize for better face detection
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=4,
    minSize=(60, 60)
)


        if len(faces) == 0:
            return {"error": "No face detected"}

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face = frame[y:y+h, x:x+w]

        img = Image.fromarray(face)
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            age = model(img).item()

        age = max(1, min(100, age))
        category = age_category(age)

        return {
            "age": round(age, 2),
            "age_group": category
        }

    except Exception as e:
        return {"error": str(e)}
