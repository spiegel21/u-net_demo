import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import torchvision.transforms.functional as TF

class PetDataset(Dataset):
    def __init__(self, root_dir='./data', train=True, transform=None):
        """
        Args:
            root_dir (string): Directory with the Oxford Pet dataset
            train (bool): Whether to use train or test set
            transform (callable, optional): Optional transform to be applied
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Check if dataset is already downloaded
        dataset_exists = self.is_dataset_downloaded()

        # Download and load the Oxford-IIIT Pet Dataset
        self.dataset = datasets.OxfordIIITPet(
            root=root_dir,
            split='trainval' if train else 'test',
            target_types='segmentation',
            download=not dataset_exists  # Download only if not already downloaded
        )
    
    def is_dataset_downloaded(self):
        # Check if the images directory exists
        images_dir = os.path.join(self.root_dir, 'oxford-iiit-pet', 'images')
        return os.path.exists(images_dir)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        
        # Convert mask to binary (foreground/background)
        mask = np.array(mask)
        mask = (mask > 0).astype(np.float32)
        
        # Convert to tensors and normalize
        image = TF.to_tensor(image)
        mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension
        
        # Resize to a fixed size
        image = TF.resize(image, (256, 256))
        mask = TF.resize(mask, (256, 256))
        
        # Normalize image
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        sample = {
            'image': image,
            'mask': mask
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

def get_dataloaders(base_dir='./data', batch_size=4):
    """
    Creates training and validation dataloaders
    """
    # Create datasets
    train_dataset = PetDataset(root_dir=base_dir, train=True)
    val_dataset = PetDataset(root_dir=base_dir, train=False)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader