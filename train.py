import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Precision, Recall, F1
from dataset import get_dataloaders
from model import ViT
import os

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 10
log_dir = './logs'
weights_dir = './weights'

# Create weights directory if it doesn't exist
os.makedirs(weights_dir, exist_ok=True)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir)

# Load data
train_loader, val_loader = get_dataloaders(root="./dataset", batch_size=batch_size)

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViT().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

precision_metric = Precision(average='weighted').to(device)
recall_metric = Recall(average='weighted').to(device)
f1_metric = F1(average='weighted').to(device)

def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/train', epoch_acc, epoch)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

def validate(model, val_loader, criterion, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            precision_metric.update(predicted, labels)
            recall_metric.update(predicted, labels)
            f1_metric.update(predicted, labels)

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()
    f1 = f1_metric.compute().item()

    writer.add_scalar('Loss/val', epoch_loss, epoch)
    writer.add_scalar('Accuracy/val', epoch_acc, epoch)
    writer.add_scalar('Precision/val', precision, epoch)
    writer.add_scalar('Recall/val', recall, epoch)
    writer.add_scalar('F1_Score/val', f1, epoch)

    print(f"Validation Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

def save_checkpoint(model, optimizer, epoch, directory):
    checkpoint_path = os.path.join(directory, f'epoch_{epoch+1}.pth')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

if __name__ == "__main__":
    for epoch in range(num_epochs):
        train(model, train_loader, criterion, optimizer, epoch)
        save_checkpoint(model, optimizer, epoch, weights_dir)
        validate(model, val_loader, criterion, epoch)

    writer.close()