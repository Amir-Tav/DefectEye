import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Initialize the model and move to GPU
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Measure GPU memory before training
torch.cuda.reset_max_memory_allocated(device)
gpu_memory_before = torch.cuda.memory_allocated(device) / (1024**2)  # Convert bytes to MB

# Train the model
num_epochs = 2
start_time = time.time()
for epoch in range(num_epochs):
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

end_time = time.time()
gpu_memory_after = torch.cuda.memory_allocated(device) / (1024**2)  # Convert bytes to MB

# Print training time and GPU memory usage
print(f"Training completed in {end_time - start_time:.2f} seconds")
print(f"GPU Memory Usage: {gpu_memory_after - gpu_memory_before:.2f} MB")

# Save the trained model
torch.save(model.state_dict(), "simple_nn.pth")
print("Model saved as simple_nn.pth")
