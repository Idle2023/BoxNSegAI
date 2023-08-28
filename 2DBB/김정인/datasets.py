import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform

class CustomDataset(Dataset):
    def __init__(self, data_folder, split):
        self.split = split.lower()
        assert self.split in {'training', 'test'}

        self.data_folder = data_folder

        # Read data files
        # with open('/Users/jeongin/PycharmProjects/2dbb_ji/json_files/training_images.json') as j:
        with open(os.path.join(data_folder, 'json_files', self.split +'_images.json'),'r') as j:
            self.images = json.load(j)
            # training_images.json 에 이미지 파일 제목이 리스트에 전부 다 들어 있음
        # with open('/Users/jeongin/PycharmProjects/2dbb_ji/json_files/training_objects.json') as j:
        with open(os.path.join(data_folder, 'json_files', self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r') # 제목 리스트에서 하나씩 불러 와서 PIL.Image 로 읽음
        image = image.convert('RGB')
        # Read objects in this image (bounding boxes, labels)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes']) # 좌표값 텐서로 변환 (실수형)
        labels = torch.LongTensor(objects['labels']) # 레이블 텐서로 변환 (정수형)
        # Apply transformations
        image, boxes, labels = transform(image, boxes, labels, split=self.split)
        return image, boxes, labels

    def collate_fn(self, batch):

        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)
        return images, boxes, labels # tensor (N, 3, 300, 300)
