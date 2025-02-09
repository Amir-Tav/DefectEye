import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Disable XNNPACK (if it exists)
if hasattr(torch.backends, 'xnnpack'):
    torch.backends.xnnpack.enabled = False

# Path to the TorchScript model
MODEL_PATH = "D:/Coding Projects/DefectEye/runs/detect/updated-weights3/weights/best.torchscript"

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the TorchScript model correctly
model = torch.jit.load(MODEL_PATH, map_location=device)
model.to(device)
model.eval()

# Recompile model to remove XNNPACK dependency
model = torch.jit.optimize_for_inference(model)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Convert frame to PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Preprocess image
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)

    # Check if the model output is compatible (modify based on your model type)
    if isinstance(output, torch.Tensor):
        pred = torch.argmax(output, dim=1).item()
    else:
        pred = output  # Adjust this if your model returns something different

    # Display prediction on frame
    cv2.putText(frame, f"Prediction: {pred}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show frame
    cv2.imshow('Real-Time Prediction', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
