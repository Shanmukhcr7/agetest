# Age-X: Next-Generation Age Verification Platform

![Status](https://img.shields.io/badge/Status-Production%20Ready-success)
![Security](https://img.shields.io/badge/Security-Safety%20Critical-blue)
![Stack](https://img.shields.io/badge/Stack-FastAPI%20%7C%20PyTorch%20%7C%20VanillaJS-orange)

## Executive Summary
Age-X is a safety-critical, high-performance web application designed to protect minors from age-inappropriate content in real-time. Unlike traditional age gates, Age-X utilizes on-device camera input and server-side neural inference (ResNet18) to continuously monitor the viewer's age group and filter content dynamically.

This repository contains the production-ready source code for the platform, engineered with a "Safety-First" architecture where fallback states always default to the most restrictive content category (Kid Mode).

---

## 🏗️ Architecture

### High-Level Design
The system follows a decoupled Client-Server architecture:
1.  **Frontend (Vanilla JS + GSAP)**: A lightweight, accessible client that manages the camera feed, handles secure video playback, and renders a glassmorphism-based UI.
2.  **Backend (FastAPI)**: A stateless, high-throughput REST API that provides inference services using a loaded PyTorch model.
3.  **Safety Engine**: Both client-side and server-side logic enforce "Strict Unlocking", where upgrading to Adult privileges requires sustained high-confidence verification, while downgrading to Kid mode is instantaneous.

### Directory Structure
```
Age-X/
├── backend/
│   ├── app.py              # Application Entry Point (FastAPI)
│   ├── age_service.py      # Core Inference Logic & Model Singleton
│   ├── age_model.pth       # Neural Network Weights (ResNet18)
│   └── requirements.txt    # Python Dependencies
├── frontend/
│   ├── index.html          # Semantic Entry Point
│   ├── style.css           # Premium CSS / Glassmorphism System
│   ├── app.js              # Application Logic (Safety Engine, Reel Manager)
│   └── firebase.js         # Content Database Config
└── README.md               # Documentation
```

---

## 🔒 Security & Privacy Protocol

### Data Handling
- **Ephemeral Processing**: Images sent to `/api/age-check` are processed in memory and typically discarded after inference. We do not store face data on disk.
- **No PII Storage**: No facial embeddings or personally identifiable information is persisted.
- **HTTPS Enforcement**: In production, all API calls must be over TLS 1.3.

### Safety Logic (Strict Mode)
The `SafetyEngine` implements a state machine with the following rules:
- **Default State**: `Kid` (Locked).
- **Upgrade Path**: `Kid` -> `Adult` requires **5 consecutive frames** of >85% confidence.
- **Downgrade Path**: `Adult` -> `Kid` occurs on **1 single frame** of detection (Zero Tolerance).
- **Fail-Safe**: Network errors, camera failures, or ambiguous results force `Kid` mode.

---

## 🚀 Setup & Deployment

### Prerequisities
- Python 3.10+
- Node.js (optional, for local serving)
- Webcam

### Installation
1.  **Backend Setup**:
    ```bash
    cd backend
    pip install -r requirements.txt
    ```
2.  **Start API**:
    ```bash
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
    ```
3.  **Frontend Setup**:
    - Serve the `frontend` directory using any static file server (e.g., Live Server in VS Code):
    ```bash
    cd frontend
    npx serve .
    ```

### Model configuration
Ensure `age_model.pth` is present in `backend/`. If missing, the system will initialize but fail open to Safety Mode (Kid) for all requests. The model should be a standard ResNet18 trained for age regression.

---

## 🧪 Performance Targets
- **Inference Latency**: ~80-150ms on CPU (optimized with `opencv-python-headless`).
- **UI Interaction**: 60fps animations via GSAP.
- **Responsiveness**: Mobile-first design supporting viewports from 320px to 4K.

---

## ⚖️ License & Compliance
Proprietary software tailored for silicon-valley standard safety compliance.
