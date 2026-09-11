import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# 1. Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. Load Dataset from folders
train_dataset = datasets.ImageFolder(root="dataset/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 3. Setup Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = models.MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights)

num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 2)
model = model.to(device)

# 4. Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Quick Training Loop (5 epochs)
print("Training model...")
model.train()
for epoch in range(5):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/5 completed. Loss: {running_loss/len(train_loader):.4f}")

# 6. Save the trained weights
torch.save(model.state_dict(), "pet_classifier_mobilenetv2.pth")
print("Model trained and saved as pet_classifier_mobilenetv2.pth successfully!")