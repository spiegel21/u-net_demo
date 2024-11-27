import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from datetime import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import PetDataset, get_dataloaders
from unet import UNet

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_dir='checkpoints'):
    """
    Training function for U-Net model
    """
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch in train_bar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix(loss=f'{loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for batch in val_bar:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_bar.set_postfix(loss=f'{loss.item():.4f}')
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Average Training Loss: {avg_train_loss:.4f}')
        print(f'Average Validation Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
    
    return train_losses, val_losses

def plot_losses(train_losses, val_losses, save_dir='plots'):
    """
    Plot training and validation losses
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    plt.close()

def visualize_predictions(model, val_loader, device, num_samples=3, save_dir='predictions'):
    """
    Visualize model predictions
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_samples:
                break
                
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            outputs = model(images)
            
            # Convert to binary predictions
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            
            # Plot results
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Convert tensor to numpy for visualization
            img_np = images[0].cpu().permute(1, 2, 0).numpy()
            # Denormalize image
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
            
            axes[0].imshow(img_np)
            axes[0].set_title('Input Image')
            axes[1].imshow(masks[0, 0].cpu().numpy(), cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[2].imshow(predictions[0, 0].cpu().numpy(), cmap='gray')
            axes[2].set_title('Prediction')
            
            # Remove axes
            for ax in axes:
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'prediction_{i+1}.png'))
            plt.close()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Configuration
    BATCH_SIZE = 8  # Increased batch size for faster training
    NUM_EPOCHS = 30  # Reduced epochs as we have more data
    LEARNING_RATE = 3e-4  # Slightly increased learning rate
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create data loaders
    train_loader, val_loader = get_dataloaders(batch_size=BATCH_SIZE)
    
    # Initialize model (3 input channels for RGB images)
    model = UNet(n_channels=3, n_classes=1, bilinear=True)
    model = model.to(DEVICE)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'runs/{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    # Train model
    print(f"Training on device: {DEVICE}")
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        save_dir=os.path.join(save_dir, 'checkpoints')
    )
    
    # Plot losses
    plot_losses(train_losses, val_losses, save_dir=os.path.join(save_dir, 'plots'))
    
    # Load best model for visualization
    checkpoint = torch.load(os.path.join(save_dir, 'checkpoints/best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Visualize predictions
    visualize_predictions(
        model=model,
        val_loader=val_loader,
        device=DEVICE,
        save_dir=os.path.join(save_dir, 'predictions')
    )

if __name__ == '__main__':
    main()