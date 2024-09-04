from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

# Define data augmentation and preprocessing transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((144, 144)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize((144, 144)),
        transforms.ToTensor(),
    ]),
}

def get_dataloaders(root, batch_size=32):
    # Load the trainval dataset
    train_dataset = OxfordIIITPet(root=root, split='trainval', download=True, transform=data_transforms['train'])
    
    # Load the test dataset as validation dataset
    val_dataset = OxfordIIITPet(root=root, split='test', download=True, transform=data_transforms['val'])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def show_images(images, num_samples=40, cols=8):
    plt.figure(figsize=(15, 15))
    idx = int(len(images) / num_samples)
    for i, img in enumerate(images):
        if i % idx == 0:
            plt.subplot(int(num_samples / cols) + 1, cols, int(i / idx) + 1)
            plt.imshow(to_pil_image(img[0]))
    plt.show()

# Example usage
if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(root=".")
    # Display some training images
    sample_images = next(iter(train_loader))
    show_images(sample_images[0])