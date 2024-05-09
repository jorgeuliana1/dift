import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import random
import time
import torch.nn as nn
import os

from src.models.dift_sd import SDFeaturizer
from src.models.dift_sd import ShallowNetwork
from src.models.dift_sd import SDClassifier
from src.utils.dataloader import get_dataset

# Function to evaluate the model on the validation set
def evaluate_classifier(classifier, validation_loader, device):
    classifier.featurizer.pipe.to(device)
    classifier.shallow_network.to(device)

    total = 0
    correct = 0
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = classifier.classify(images)
            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return total_loss / len(validation_loader), accuracy


# Function to train the SDClassifier with validation and checkpointing
def train_classifier(
    classifier: SDClassifier,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
    lr: float = 1e-3,
    checkpoint_path: str = "./best_model.pth"
):
    optimizer = optim.Adam(classifier.shallow_network.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    classifier.featurizer.pipe.to(device)
    classifier.shallow_network.to(device)

    best_accuracy = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits = classifier.classify(images)
            loss = loss_fn(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        # Calculate training accuracy
        train_accuracy = correct / total
        epoch_duration = time.time() - start_time
        
        # Evaluate on the validation set
        val_loss, val_accuracy = evaluate_classifier(classifier, validation_loader, device)

        # Save the best-performing model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(classifier.shallow_network.state_dict(), checkpoint_path)
        
        print(
            f"Epoch {epoch + 1}/{num_epochs}: Train Loss = {total_loss / len(train_loader):.4f}, "
            f"Train Accuracy = {train_accuracy:.4f}, Validation Loss = {val_loss:.4f}, "
            f"Validation Accuracy = {val_accuracy:.4f}, Time = {epoch_duration:.2f}s"
        )


# Main function with validation and checkpointing
def main():
    featurizer = SDFeaturizer()  # Adjust if needed with prompts
    classifier = ShallowNetwork(224 * 224 * 3, 256, 6)
    
    model = SDClassifier(featurizer, classifier)
    train_dataset, test_dataset = get_dataset(
        "PAD-UFES-20",
        "dataroot/PAD-UFES-20",
        "./dataset/PAD-UFES-20/pad-ufes-20_parsed_folders.csv",
        "./dataset/PAD-UFES-20/pad-ufes-20_parsed_test.csv"
    )
    
    train_loader = DataLoader(train_dataset, batch_size=9, num_workers=6, shuffle=True)
    validation_loader = DataLoader(test_dataset, batch_size=9, num_workers=6, shuffle=True)

    # Ensure correct device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Train the model with checkpointing
    train_classifier(
        model,
        train_loader,
        validation_loader,
        1000,
        device,
        lr=1e-3,
        checkpoint_path="./best_model.pth"
    )

if __name__ == "__main__":
    main()
