import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import torchvision
import json
import os
from PIL import Image
from tqdm import tqdm
# from utils import transform

class Dataset(Dataset):
    # A Pytorch dataset class to be used in a Pytorch DataLoader to create batches.
    def __init__(self, data_folder, split, transform=None):
        self.data_folder = data_folder
        self.split = split
        self.image_folder = os.path.join(data_folder, split, "images")
        self.label_folder = os.path.join(data_folder, split, "labels")
        self.transform = transform
        self.image_files = sorted([f for f in tqdm(os.listdir(self.image_folder), desc = "Loading...") if f.endswith('.jpg')])

        self.label_map = {
            "car": 0,
            "truck": 1,
            "bus": 2,
            "special_vehicle": 3,
            "motorcycle": 4,
            "bicycle": 5,
            "pedestrian": 6,
            "traffic_sign": 7,
            "traffic_light": 8,
            "none": 9
        }
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        label_path = img_path.replace("image", "labels").replace(".jpg", ".json")
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        else:
            image = ToTensor()(image)

        with open(label_path, 'r') as file:
            label_data = json.load(file)
        boxes = [anno['Coordinate'] for anno in label_data['Annotation']]
        labels = [anno['Label'] for anno in label_data['Annotation']]
        labels = [self.label_map[label] for label in labels]

        target = {}
        # box = [x_min, y_min, width, height] -> [x_min, y_min, x_max, y_max]
        converted_boxes = [[box[0], box[1], box[0]+box[2], box[1]+box[3]] for box in boxes]
        target["boxes"] = torch.tensor(converted_boxes, dtype = torch.float32) # 좌표값을 텐서에 저장한 것
        target["labels"] = torch.tensor(labels, dtype = torch.int64) # label (0~9) 를 텐서로 저장한 것
        return image, target