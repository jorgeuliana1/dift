import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import random
import time
import torch.nn as nn

from src.models.dift_sd import SDFeaturizer
from src.models.dift_sd import ShallowNetwork
from src.models.dift_sd import SDClassifier
from src.utils.dataloader import get_dataset

# Function to train the SDClassifier
def train_classifier(
    classifier: SDClassifier,
    train_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
    lr: float = 1e-3
):
    # Optimizer
    optimizer = optim.Adam(classifier.shallow_network.parameters(), lr=lr)
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Move classifier to the correct device
    classifier.featurizer.pipe.to(device)
    classifier.shallow_network.to(device)

    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            # Move data to the correct device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass and compute loss
            logits = classifier.classify(images)
            loss = loss_fn(logits, labels)
            
            # Backward pass and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate loss and accuracy
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        # Calculate accuracy for the epoch
        accuracy = correct / total
        epoch_duration = time.time() - start_time
        
        print(f"Epoch {epoch + 1}/{num_epochs}: Loss = {total_loss / len(train_loader):.4f}, Accuracy = {accuracy:.4f}, Time = {epoch_duration:.2f}s")

def main():
    featurizer = SDFeaturizer() # Can improve: adding prompt
    classifier = ShallowNetwork(224 * 224 * 3, 256, 6)
    
    model = SDClassifier(featurizer, classifier)
    train_dataset, test_dataset = get_dataset(
        "PAD-UFES-20",
        "dataroot/PAD-UFES-20",
        "./dataset/PAD-UFES-20/pad-ufes-20_parsed_folders.csv",
        "./dataset/PAD-UFES-20/pad-ufes-20_parsed_test.csv"
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=9,
        num_workers=6,
        shuffle=True,
    )
    
    train_classifier(model, train_loader, 1000, "cuda:0")

if __name__ == "__main__":
    main()