from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import transforms
import io
import logging
import uvicorn


# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO)

# Import torch and model-related components
import torch
import torch.nn as nn

# Define the CNN model class
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 16)  # Predicts a single number: count of people

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

app = FastAPI()

# Instantiate the model and load state dict
model = CNNModel()
model.load_state_dict(torch.load('model_state_dict.pth'))
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Helper function to process and predict the image
def process_and_predict_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image_tensor)
            predicted_count = torch.round(output).item()
        return predicted_count
    except Exception as e:
        logging.error(f"Failed to process and predict image: {str(e)}")
        raise

# Define the endpoint for prediction
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        predicted_count = process_and_predict_image(image_bytes)
        return JSONResponse(content={"predicted_count": predicted_count})
    except Exception as e:
        logging.error(f"Error in /predict/ endpoint: {str(e)}")
        return JSONResponse(status_code=500, content={"message": "Error processing the image"})

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
