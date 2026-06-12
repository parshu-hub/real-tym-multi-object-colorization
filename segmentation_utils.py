import cv2
import torch
import numpy as np
from torchvision import models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.segmentation.deeplabv3_resnet101(weights="DEFAULT")
model.eval()
model.to(device)

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((360, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

CLASS_COLORS = {
    0: (0, 0, 0),
    6: (255, 0, 0),
    7: (0, 0, 255),
    15: (0, 255, 255),
    16: (0, 255, 0),
    19: (255, 255, 0)
}

def create_color_mask(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        color_mask[mask == class_id] = color
    return color_mask

def segment_and_colorize(frame):
    original_h, original_w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)["out"][0]
    prediction = output.argmax(0).byte().cpu().numpy()

    color_mask = create_color_mask(prediction)
    color_mask = cv2.resize(color_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    blended = cv2.addWeighted(frame, 0.6, color_mask, 0.4, 0)

    return blended, color_mask