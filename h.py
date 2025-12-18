import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time

# ================= CONFIG =================
MODEL_PATH = "age_model.pth"
IMG_SIZE = 224
STABLE_FRAMES = 5   # 🔥 number of frames to average
# =========================================


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


def main():

    # -------- Device --------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

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

    # -------- Webcam --------
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Camera not accessible")
        return

    print("📸 Press SPACE to capture stable prediction")
    print("❌ Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        cv2.imshow("Camera - Stable Age Test", frame)

        key = cv2.waitKey(1) & 0xFF

        # -------- Stable Capture --------
        if key == 32:  # SPACE
            ages = []
            print("⏳ Capturing frames for stabilization...")

            for i in range(STABLE_FRAMES):
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(
                    gray, 1.2, 5, minSize=(100, 100)
                )

                if len(faces) == 0:
                    print(f"Frame {i+1}: No face detected")
                    time.sleep(0.2)
                    continue

                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face = frame[y:y+h, x:x+w]
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                img = Image.fromarray(face_rgb)
                img = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    age = model(img).item()

                age = max(1, min(100, age))
                ages.append(age)

                print(f"Frame {i+1}: Age = {age:.2f}")
                time.sleep(0.2)  # small delay

            if len(ages) == 0:
                print("❌ No valid frames captured. Try again.")
                continue

            avg_age = sum(ages) / len(ages)
            category = age_category(avg_age)

            print("\n✅ STABLE RESULT")
            print(f"Average Age: {avg_age:.2f}")
            print(f"Final Category: {category}\n")

            # Display result
            cv2.putText(
                frame,
                f"{int(avg_age)} yrs | {category}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3
            )
            cv2.imshow("Stable Prediction Result", frame)

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
