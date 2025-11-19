import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# -------------------------
# Hyperparameters
# -------------------------
BATCH_SIZE = 32
LR = 0.0015
EPOCHS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Transforms / Datasets / Loaders
# -------------------------

train_transform = transforms.Compose([

    # Randomly rotates images by 13 degrees
    transforms.RandomRotation(13),
    # Randomly crops the image
    transforms.RandomCrop(28, padding=1),
    # Transforms image to a PyTorch tensor and changes shape to (1, 28, 28)
    transforms.ToTensor()
])

# No augmentations in test transform
# You only want the model evaluated on clean, unmodified data.
test_transform = transforms.Compose([
    transforms.ToTensor()
])
train_dataset = datasets.MNIST(
    root="data", train=True, download=True, transform=train_transform
)
test_dataset = datasets.MNIST(
    root="data", train=False, download=True, transform=test_transform
)

# Setting it to test on only 30% of the data with a fixed seed for reproducibility
torch.manual_seed(13)
train_size = int(0.3 * len(train_dataset))
_, small_train_dataset = random_split(train_dataset, [len(train_dataset) - train_size, train_size])


# Loads the smaller dataset, shuffled
train_loader = DataLoader(small_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# Loads the full dataset when testing
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Subset training samples: {len(small_train_dataset)}")

# -------------------------
# Model (CNN architecture)
# -------------------------
class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # These are the conv / relu / pool layers
            # Input: 1 x 28 x 28

            # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
            # This one takes in one channel (since MNIST images are grayscale)
            # And extracts 16 features.
            # Output: 18 x 28 x 28
            nn.Conv2d(1, 18, 3, stride=1, padding=1),

            #nn.BatchNorm2d(num_features). Normalization to stabilize training
            nn.BatchNorm2d(18),

            # Activation function to add nonlinearity
            nn.ReLU(),

            # nn.MaxPool2d(kernel_size, stride)
            # Pooling layer. Shrinks spatial dimensions to condense info
            # This one halves size
            # Output: 18 x 14 x 14
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            # These are the Flatten(), Linear(), etc. layers

            # Outputs 18*14*14 = 3528. Needed prep for linear (fully connected) layers
            nn.Flatten(),
            nn.Linear(3528, 128),
            nn.ReLU(),
            # 50% chance that ReLU will be dropped. Helps avoid overfitting
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            # Outputs to logits for digits 0-9
            nn.Linear(64, 10)
        )

# Forward pass
    def forward(self, x):

        # Takes the input image, runs through all layers in self.features
        # Outputs a feature map
        x = self.features(x)

        # Takes the feature map, runs it through the classifier layers,
        # And outputs logits for the digits 0-9
        x = self.classifier(x)
        return x
    
model = MyCNN().to(device)

# -------------------------
# Loss / Optimizer
# -------------------------

# Loss function. Applies log_softmax to model outputs to convert them to probabilities
criterion = nn.CrossEntropyLoss()

# Penalizes network (big loss if it makes wrong choice with large confidence)
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------------------------
# Training Function
# -------------------------
def train_one_epoch():
    model.train()
    total_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# -------------------------
# Evaluation Function
# -------------------------
def evaluate():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    return acc

# -------------------------
# Scheduler
# -------------------------

# Every 2 epochs, LR is multiplied by gamma (cut by 33%)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma = 0.67)

# -------------------------
# Main Training Loop
# -------------------------
for epoch in range(EPOCHS):
    loss = train_one_epoch()
    acc = evaluate()

    scheduler.step()

    # Converts to percentage
    acc = acc * 100
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss {loss:.4f} | Test Accuracy {acc:.2f}%")