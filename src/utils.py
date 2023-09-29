import os

import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2


def draw_mask(image, mask):
    color_mask = np.zeros_like(image)
    color_mask[:, :, 1] += mask.astype('uint8') * 250
    masked = cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)

    return masked

def save_images(model, dataloader, save_dir, device):
    model.eval()
    for idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            outputs = torch.argmax(outputs, dim=1)

        for i in range(len(inputs)):
            input_image = inputs[i].detach().cpu().permute(1, 2, 0).numpy()
            output_image = outputs[i].detach().cpu().numpy()

            # Create the figure and subplots
            fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=100)
            fig.subplots_adjust(hspace=0.1, wspace=0.1)
            
            # Plot input image
            axes[0].imshow(input_image)
            axes[0].set_title("Input")
            axes[0].axis('off')
            
            # Plot segmentation output
            axes[1].imshow(draw_mask(input_image, output_image))
            axes[1].set_title("Road Segmentation")
            axes[1].axis('off')
            
            # Save the figure
            save_path = os.path.join(save_dir, f'figure_{idx}_{i}.png')
            plt.savefig(save_path)
            
            plt.close(fig)