import os
import time
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Path to the dataset
dataset_path = os.path.join(os.path.expanduser("~"), ".cache", "kagglehub", 
                           "datasets", "yusufberksardoan", "traffic-detection-project", 
                           "versions", "1")

# Only vehicle classes (no person)
class_names = ["bicycle", "bus", "car", "motorbike"]
num_classes = len(class_names)
print(f"Training model with {num_classes} vehicle classes: {', '.join(class_names)}")

# Class ID mapping (YOLO dataset uses these IDs)
class_id_mapping = {
    0: 0,  # bicycle -> 0
    1: 1,  # bus -> 1
    2: 2,  # car -> 2
    3: 3,  # motorbike -> 3
    # 4 (person) is excluded
}

# Vehicle dataset class
class VehicleDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, max_per_class=100, max_images=300):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.max_per_class = max_per_class
        self.max_images = max_images
        
        self.images_dir = os.path.join(root_dir, split, 'images')
        self.labels_dir = os.path.join(root_dir, split, 'labels')
        
        # Load all image filenames
        self.image_files = sorted([f for f in os.listdir(self.images_dir) 
                               if f.endswith('.jpg')])[:self.max_images]
        
        self.items = self._extract_objects()
    
    def _extract_objects(self):
        items = []
        class_counts = {i: 0 for i in range(num_classes)}
        
        print(f"Processing {len(self.image_files)} images for {self.split}...")
        for img_file in self.image_files:
            # Break if we have enough samples for each class
            if all(count >= self.max_per_class for count in class_counts.values()):
                break
                
            # Get corresponding label file
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(self.labels_dir, label_file)
            
            if not os.path.exists(label_path):
                continue
                
            # Load image
            img_path = os.path.join(self.images_dir, img_file)
            try:
                image = Image.open(img_path)
                img_width, img_height = image.size
            
                # Parse YOLO format labels
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                            
                        original_class_id = int(parts[0])
                        
                        # Skip if not a vehicle class (person=4)
                        if original_class_id == 4:
                            continue
                            
                        # Map to new class index (0-3)
                        class_id = class_id_mapping[original_class_id]
                        
                        # Skip if we have enough samples for this class
                        if class_counts[class_id] >= self.max_per_class:
                            continue
                            
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert to pixel coordinates
                        x1 = int((x_center - width/2) * img_width)
                        y1 = int((y_center - height/2) * img_height)
                        x2 = int((x_center + width/2) * img_width)
                        y2 = int((y_center + height/2) * img_height)
                        
                        # Ensure coordinates are within bounds
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(img_width, x2)
                        y2 = min(img_height, y2)
                        
                        # Skip small objects
                        if x2 - x1 < 20 or y2 - y1 < 20:
                            continue
                        
                        try:
                            # Crop the object directly
                            crop = image.crop((x1, y1, x2, y2))
                            
                            # Add to the dataset
                            items.append((crop, class_id))
                            class_counts[class_id] += 1
                        except Exception as e:
                            print(f"Error cropping image {img_file}: {e}")
            except Exception as e:
                print(f"Error processing image {img_file}: {e}")
        
        print(f"Extracted {len(items)} vehicle objects for {self.split}")
        print(f"Class distribution: {class_counts}")
        return items
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        image, label = self.items[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Define model architecture based on EfficientNet (better than MobileNetV2 for this task)
class VehicleClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(VehicleClassifier, self).__init__()
        
        # Load pretrained EfficientNet-B0 (smaller and faster than B1-B7)
        self.model = models.efficientnet_b0(pretrained=True)
        
        # Freeze early layers
        for param in list(self.model.parameters())[:-10]:
            param.requires_grad = False
        
        # Replace the classifier
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

# Data augmentation for training (more aggressive than before)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation transformation
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets with increased size
print("Loading datasets...")
train_dataset = VehicleDataset(
    root_dir=dataset_path, 
    split='train', 
    transform=train_transform,
    max_per_class=150,  # More samples per class
    max_images=400      # Process more images
)

val_dataset = VehicleDataset(
    root_dir=dataset_path, 
    split='valid', 
    transform=val_transform,
    max_per_class=40,   # More validation samples
    max_images=200      # Process more images
)

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Create the model
model = VehicleClassifier(num_classes=num_classes)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
# Higher learning rate and weight decay to prevent overfitting
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # Reduce learning rate every 5 epochs

# Training function with improvements
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=20, patience=7):
    print("Starting training...")
    since = time.time()
    
    best_model_wts = model.state_dict()
    best_acc = 0.0
    no_improve_epochs = 0
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
                if scheduler:
                    scheduler.step()
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                # Deep copy the model if best accuracy
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict().copy()
                    # Reset no improvement counter
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
        
        # Early stopping check
        if no_improve_epochs >= patience:
            print(f'Early stopping after {patience} epochs without improvement.')
            break
        
        print()
    
    # Training time
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.legend()
    
    plt.savefig('vehicle_training_history.png')
    plt.close()
    
    return model, history

# Train the model
model, history = train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=20, patience=7)

# Save the model
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': class_names,
    'history': history
}, 'vehicle_classifier.pth')

print("Model saved to 'vehicle_classifier.pth'")

# Test dataset
test_dataset = VehicleDataset(
    root_dir=dataset_path, 
    split='test', 
    transform=val_transform,
    max_per_class=50,
    max_images=200
)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Evaluate model
def evaluate_model(model, dataloader):
    model.eval()
    
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            predictions.extend(preds.cpu().numpy())
            ground_truth.extend(labels.cpu().numpy())
            
    # Calculate overall accuracy
    correct = sum(p == gt for p, gt in zip(predictions, ground_truth))
    accuracy = correct / len(predictions)
    
    # Calculate per-class accuracy
    class_correct = {class_name: 0 for class_name in class_names}
    class_total = {class_name: 0 for class_name in class_names}
    
    for pred, gt in zip(predictions, ground_truth):
        if pred == gt:
            class_correct[class_names[gt]] += 1
        class_total[class_names[gt]] += 1
    
    class_accuracy = {cls: correct/max(1, total) for cls, correct, total in 
                     [(cls, class_correct[cls], class_total[cls]) for cls in class_names]}
    
    return accuracy, class_accuracy, predictions, ground_truth

# Evaluate model
accuracy, class_accuracy, predictions, ground_truth = evaluate_model(model, test_loader)

print(f"Test Accuracy: {accuracy:.4f}")
print("Per-class accuracy:")
for cls, acc in class_accuracy.items():
    print(f"  {cls}: {acc:.4f}")

# Create confusion matrix plot
def plot_confusion_matrix(predictions, ground_truth):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(ground_truth, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    print("Confusion matrix saved to 'confusion_matrix.png'")

plot_confusion_matrix(predictions, ground_truth)

print("Model evaluation complete.") 
