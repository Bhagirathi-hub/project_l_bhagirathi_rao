import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from config import resize_x, resize_y

class UnicornImgDataset(Dataset):
    def __init__(self, root_folder):
        self.image_paths = []
        self.labels = []
        self.transform = transforms.Compose([
            transforms.Resize((resize_x, resize_y)),
            transforms.ToTensor()
        ])

        # NEW: Now we scan subfolders like zebra/, elephant/
        for subfolder in os.listdir(root_folder):
            subfolder_path = os.path.join(root_folder, subfolder)
            if os.path.isdir(subfolder_path):
                if "zebra" in subfolder.lower():
                    label = 0
                elif "elephant" in subfolder.lower():
                    label = 1
                else:
                    raise ValueError(f"Cannot determine label for folder: {subfolder}")
                
                for img_name in os.listdir(subfolder_path):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(subfolder_path, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = self.labels[idx]
        return image, label

def unicornLoader(folder_path, batch_size=32):
    dataset = UnicornImgDataset(folder_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader
