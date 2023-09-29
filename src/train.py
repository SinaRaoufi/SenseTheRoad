import os
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from dataset import RoadDataset
from model import HandSegmentationModel
from metrics import mean_IoU, pixel_accuracy
from utils import save_images


def train(model, optimizer, criterion, n_epoch, data_loaders: dict, device):
    
    train_losses = np.zeros(n_epoch)
    val_losses = np.zeros(n_epoch)
    best_iou = 0.0

    model.to(device)

    since = time.time()

    for epoch in range(n_epoch):
        train_loss = 0.0
        train_accuracy = 0.0
        train_iou = 0.0

        model.train()
        for inputs, targets in tqdm(data_loaders['train'], desc=f'Training... Epoch: {epoch + 1}/{EPOCHS}'):
            inputs, targets = inputs.to(device).float(), targets.to(device).float()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            train_loss += loss.item()
            train_accuracy += pixel_accuracy(outputs, targets)
            train_iou += mean_IoU(outputs, targets)

            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(data_loaders['train'])
        train_accuracy = train_accuracy / len(data_loaders['train'])
        train_iou = train_iou / len(data_loaders['train'])

        with torch.no_grad():
            val_loss = 0.0
            val_accuracy = 0.0
            val_iou = 0.0
            model.eval()
            for inputs, targets in tqdm(data_loaders['validation'], desc=f'Validating... Epoch: {epoch + 1}/{EPOCHS}'):
                inputs, targets = inputs.to(device).float(), targets.to(device).float()

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_accuracy += pixel_accuracy(outputs, targets)
                val_iou += mean_IoU(outputs, targets)

            val_loss = val_loss / len(data_loaders['validation'])
            val_accuracy = val_accuracy / len(data_loaders['validation'])
            val_iou = val_iou / len(data_loaders['validation'])
        
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),
                "epoch": epoch,
                "loss": val_loss,
                "pixel_accuracy": val_accuracy,
                "iou": val_iou
            }, SAVE_MODEL_DIR)


        # save epoch losses
        train_losses[epoch] = train_loss
        val_losses[epoch] = val_loss

        print(f"Epoch [{epoch+1}/{n_epoch}]:")
        print(
            f"Train Loss: {train_loss:.4f}, Train Pixel Accuracy: {train_accuracy:.4f}, Train IOU: {train_iou:.4f}")
        print(
            f"Validation Loss: {val_loss:.4f}, Validation Pixel Accuracy: {val_accuracy:.4f}, Validation IOU: {val_iou:.4f}")
        print('-'*20)

    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Hyperparameters
    INPUT_SIZE = 256
    BATCH_SIZE = 32
    EPOCHS = 40

    # Configurations
    DATASET_DIR_ROOT = '/home/riv/Desktop/RoadSegmentation/dataset'
    SAVE_MODEL_DIR = '/home/riv/Desktop/RoadSegmentation/saved_models/model.pt'
    OUTPUT_DIR = '/home/riv/Desktop/RoadSegmentation/output_images'
    DEVICE = 'cpu'
    if torch.cuda.is_available():
        DEVICE = 'cuda:0'
    elif torch.backends.mps.is_available():
        DEVICE = 'mps'

    transforms = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE), 2),
        transforms.ToTensor()
    ])

    
    train_dataset = RoadDataset(os.path.join(DATASET_DIR_ROOT, 'Train', 'image'),
                               os.path.join(DATASET_DIR_ROOT, 'Train', 'label'),
                               transforms)
    
    validation_dataset = RoadDataset(os.path.join(DATASET_DIR_ROOT, 'Validation', 'image'),
                                    os.path.join(DATASET_DIR_ROOT, 'Validation', 'label'),
                                    transforms)
    

    data_loaders = {
        'train': DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1
        ),
        'validation': DataLoader(
            validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1
        )
    }

    
    model = HandSegmentationModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train(model, optimizer, criterion, EPOCHS, data_loaders, DEVICE)

    checkpoint = torch.load(SAVE_MODEL_DIR)
    model.load_state_dict(checkpoint['model_state_dict'])

    save_images(model, data_loaders['validation'], OUTPUT_DIR, DEVICE)

