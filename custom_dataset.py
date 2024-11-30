import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(os.listdir('./data/WildRF/train')))}
        print(self.root_dir)
        print(self.class_to_idx)
        # Collect image paths and labels
        print('root_dir', root_dir)
        if root_dir  == './data/WildRF/test':
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
                                        self.image_paths.append(img_path)
                                        self.labels.append(self.class_to_idx[subclass_name])
        else:
            print(root_dir)
            for class_name in os.listdir(root_dir):
              class_dir = os.path.join(root_dir, class_name)
              if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                  img_path = os.path.join(class_dir, img_name)
                  if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif')):
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
