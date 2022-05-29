from torch.utils.data import Dataset
import numpy as np
import torch
import os
import glob
from PIL import Image
from torchvision import transforms



class EmotionDataset(Dataset):
    def __init__(self, path):
        self.classes = []
        self.imgs_path = path
        file_list = glob.glob(self.imgs_path + "/*")
        self.data = []
        print(file_list)
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            self.classes.append(class_name)
            for img_path in glob.glob(class_path + "/*"):

                self.data.append([img_path, class_name])
        self.preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = Image.open(img_path)
        img = img.convert(mode='RGB')
        class_id = self.classes.index(class_name)
        input_tensor = self.preprocess(img)
        class_id = torch.tensor(class_id)
        return input_tensor, class_id
