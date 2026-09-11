from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI(title="Pet Image Classification API", version="1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model Architecture & Weights
print("Loading model...")
weights = models.MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 2)

try:
    model.load_state_dict(torch.load("pet_classifier_mobilenetv2.pth", map_location=device))
    print("Model weights loaded successfully!")
except Exception as e:
    print(f"Error loading model weights: {e}")

model = model.to(device)
model.eval()

class_names = ["cat", "dog"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/")
def home():
    return {"message": "Pet Classifier API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), expected_category: str = None):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded is not an image.")
    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
            
        predicted_label = class_names[predicted_idx.item()]
        confidence_score = confidence.item() * 100

        mismatch = False
        if expected_category and expected_category.lower() in ["cat", "dog"]:
            if predicted_label != expected_category.lower():
                mismatch = True

        return {
            "predicted_pet": predicted_label,
            "confidence": round(confidence_score, 2),
            "mismatch": mismatch,
            "message": f"AI detected a {predicted_label} ({round(confidence_score, 2)}% confidence)."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))