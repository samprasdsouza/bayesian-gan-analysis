# apply gausian blur filter on the images in the folder and save them in different folder

import os
from PIL import Image
import re
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


def load_image(image_path):
    # Load image and convert to RGB
    img = Image.open(image_path).convert('RGB')
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(img).unsqueeze(0)  # Add batch dimension

def gaussian_kernel(size, sigma):
    x = torch.arange(-size // 2 + 1, size // 2 + 1)
    y = torch.arange(-size // 2 + 1, size // 2 + 1)
    xx, yy = torch.meshgrid(x, y)
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()

def gaussian_blur(image, kernel_size=3, sigma=0.5):
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)
    padding = kernel_size // 2
    blurred = F.conv2d(image, kernel, groups=3, padding=padding)
    return blurred

def extract_noise_residual(image):
    # Apply a denoising filter (Gaussian blur)
    denoised = gaussian_blur(image, kernel_size=3, sigma=0.5)
    # Subtract denoised image from original to get noise residual
    residual = image - denoised
    return residual

def tensor_to_image(tensor):
    """Convert a tensor to a PIL Image."""
    # Clamp tensor values to be in the range [0, 1]
    tensor = torch.clamp(tensor, 0, 1)

    # Remove the batch dimension and convert to numpy array
    image = tensor.squeeze().permute(1, 2, 0).numpy()

    # Convert to PIL Image
    return Image.fromarray((image * 255).astype('uint8'))

def get_images_with_pattern(root_folder, pattern=r'\d{4}\.jpeg'):
    image_paths = []
    regex = re.compile(pattern)
    
    # Walk through the root folder and its subfolders
    root_dir = './data/WildRF/test'
    for class_name in os.listdir(root_dir):
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if (os.path.isdir(class_dir)):
                for subclass_name in os.listdir(class_dir):
                    subclass_dir = os.path.join(class_dir, subclass_name)
                    if os.path.isdir(subclass_dir):
                        for img_name in os.listdir(subclass_dir):
                            img_path = os.path.join(subclass_dir, img_name)
                            if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif')):
                                image_paths.append(img_path)
    # for dirpath, _, filenames in os.walk(root_folder):
    #     for file in filenames:
    #         if regex.match(file):  # Match the file name to the pattern
    #             image_paths.append(os.path.join(dirpath, file))
    
    return image_paths

def apply_gaussian_filter(image, sigma=2):
    """Apply Gaussian filter to an image."""
    image1 = Image.open(image)
    image = load_image(image)
    fingerprint = extract_noise_residual(image)
    fingerprint_image = tensor_to_image(fingerprint)
   
    # Example: Convert fingerprint to numpy array for visualization
    fingerprint_np = fingerprint.squeeze().permute(1, 2, 0).cpu().numpy()

    # Normalize the fingerprint for better visualization
    fingerprint_np = (fingerprint_np - fingerprint_np.min()) / (fingerprint_np.max() - fingerprint_np.min())

    # Load images using PIL (replace with your image paths)
    
    image2 = fingerprint_image

    # Create a figure and axis for displaying images
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    # Display the first image
    axes[0].imshow(image1)
    axes[0].axis('off')  # Turn off axis labels
    axes[0].set_title('Image 1')  # Title for the first image

    # Display the second image
    axes[1].imshow(image2)
    axes[1].axis('off')  # Turn off axis labels
    axes[1].set_title('Image 2')  # Title for the second image

    # Show the plot
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()
    return fingerprint_image

def save_filtered_images(image_paths, save_folder, sigma=2):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)  # Create the folder if it doesn't exist
    
    for path in image_paths:
        try:
            path = './sampras-testing.jpg'
            img = Image.open(path)  # Load image
            filtered_img = apply_gaussian_filter(path, sigma=sigma)  # Apply Gaussian filter
            
            # Save the filtered image
            file_name = os.path.basename(path)
            save_path = os.path.join(save_folder, file_name)
            filtered_img.save(save_path)
            print(f"Saved filtered image: {save_path}")
        
        except Exception as e:
            print(f"Failed to process image {path}: {e}")

# Example usage
root_folder = './data/WildRF'  # Change to your folder path
save_folder = './data/WildRf-Transform/test'  # Change to your desired output folder
image_paths = get_images_with_pattern(root_folder)

#path of test folder
# Apply Gaussian filter and save the images
save_filtered_images(image_paths, save_folder, sigma=2)
