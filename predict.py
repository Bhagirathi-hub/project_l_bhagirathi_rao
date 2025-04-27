import torch
from torchvision import transforms
from PIL import Image
from config import resize_x, resize_y
from model import UnicornNet

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and weights
model = UnicornNet()
model.load_state_dict(torch.load("checkpoints/final_weights.pth", map_location=device))
model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((resize_x, resize_y)),
    transforms.ToTensor()
])

def classify_animals(list_of_img_paths):
    images = []
    for img_path in list_of_img_paths:
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        images.append(img)

    batch = torch.stack(images).to(device)

    with torch.no_grad():
        outputs = model(batch)
        _, preds = torch.max(outputs, 1)

    preds = preds.cpu().numpy().tolist()
    labels = ["zebra" if l == 0 else "elephant" for l in preds]

    return labels
