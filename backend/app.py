from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import time
import uvicorn
import logging
from age_service import AgeService

# Setup Logging
logger = logging.getLogger("uvicorn")

# Initialize App
app = FastAPI(
    title="Age-X Safety API",
    description="High-performance age detection for secure content delivery.",
    version="1.0.0"
)

# Security & CORS
# In production, specific origins should be allow-listed.
ORIGINS = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "https://your-deployment-url.com"  # Replace with actual
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for dev/demo, strict in prod
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize Service (Singleton)
age_service = AgeService()

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding

# ... (Keep existing imports)

class EncryptedPayload(BaseModel):
    encrypted_data: str
    iv: str

# Shared Key (Must match Frontend)
SECRET_KEY = b'12345678901234567890123456789012' # 32 bytes

def decrypt_image(encrypted_b64: str, iv_b64: str) -> bytes:
    try:
        # Decode Base64 components
        encrypted_bytes = base64.b64decode(encrypted_b64)
        iv_bytes = base64.b64decode(iv_b64)

        # AES CBC Decryption
        cipher = Cipher(algorithms.AES(SECRET_KEY), modes.CBC(iv_bytes), backend=default_backend())
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(encrypted_bytes) + decryptor.finalize()

        # Unpad (PKCS7)
        unpadder = padding.PKCS7(128).unpadder()
        data = unpadder.update(padded_data) + unpadder.finalize()
        
        return data
    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        raise ValueError("Decryption Error")

@app.post("/api/age-check")
async def age_check(payload: EncryptedPayload):
    try:
        # 1. Validation & Decryption
        if not payload.encrypted_data or not payload.iv:
            raise HTTPException(status_code=400, detail="Empty payload")
        
        try:
            # Decrypt to get raw Base64 Image String
            decrypted_b64_bytes = decrypt_image(payload.encrypted_data, payload.iv)
            decrypted_b64_str = decrypted_b64_bytes.decode('utf-8')
            
            # Decode Image
            image_bytes = base64.b64decode(decrypted_b64_str)
            pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
            image_np = np.array(pil_image)
            
            logger.info("Image successfully decrypted and processed.")
            
        except (ValueError, UnidentifiedImageError) as e:
            logger.error(f"Image processing error: {e}")
            raise HTTPException(status_code=400, detail="Invalid secure payload")

        # 2. Inference
        result = age_service.detect_and_predict(image_np)
        
        # 3. Handle Errors (No face, etc)
        if hasattr(result, "get") and result.get("error"):
            # Return "Kid" mode if face not found/error (Fail-Safe)
            return {
                "age_group": "Kid", 
                "confidence": 0.0, 
                "forced_safety": True,
                "msg": result.get("error")
            }

        return result

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"API Error: {e}")
        # FAIL SAFE: Always return Kid on critical error
        return {
            "age_group": "Kid", 
            "confidence": 0.0, 
            "forced_safety": True,
            "error": "Internal Processing Error"
        }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
