import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from collections import deque
import time

# ================= CONFIG =================
MODEL_PATH = "age_model.pth"
IMG_SIZE = 224
SMOOTHING_FRAMES = 20
# =========================================


# -------- Age Range (5-year bucket) --------
def age_to_range(age):
    age = int(round(age))
    age = max(1, min(100, age))
    low = (age // 5) * 5
    if low == 0:
        low = 1
    high = low + 5
    return f"{low}-{high}"


# -------- Age Category Rules --------
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


def main():

    # -------- Device (Windows Compatible) --------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("DEVICE: CUDA GPU")
    else:
        device = torch.device("cpu")
        print("DEVICE: CPU")

    # -------- Load Model (Regression) --------
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

    # -------- Age Smoothing --------
    age_buffer = deque(maxlen=SMOOTHING_FRAMES)

    # -------- Webcam (DirectShow for Windows) --------
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("❌ Camera not accessible")
        return

    print("Press Q to quit")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror view
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(100, 100)
        )

        # Detect ONLY ONE face (largest)
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

            face = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(face_rgb)
            img = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_age = model(img).item()

            pred_age = max(1, min(100, pred_age))
            age_buffer.append(pred_age)

            # Smooth age
            avg_age = sum(age_buffer) / len(age_buffer)

            # Apply rules
            age_range = age_to_range(avg_age)
            category = age_category(avg_age)

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Display results
            cv2.putText(
                frame,
                f"{age_range} | {category}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        # -------- FPS Counter --------
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow("Live Age Prediction (Windows)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
