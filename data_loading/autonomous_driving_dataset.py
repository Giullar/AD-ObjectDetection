import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont


def parse_annotations_file(file_path):
    with open(file_path, 'r') as f:
        obj_labels = []
        obj_bbs = []
        for bb in f:
            label_id, center_x, center_y, width, height = bb.rstrip().split(" ")
            obj_labels.append(int(label_id))
            obj_bbs.append([float(center_x), float(center_y), float(width), float(height)])
    return obj_labels, obj_bbs

def convert_bbs_from_yolo_to_retina(bbs_list, image_width, image_height):
    # convert from normalized [center_x, center_y, width, height] format used by yolo
    # to [x_min, y_min, x_max, y_max] format used by RetinaNet, with values between 0 and W and 0 and H.
    bbs_list_converted = []
    for bb in bbs_list:
        center_x, center_y, width, height = bb
        # de-normalize values
        center_x = center_x * image_width
        center_y = center_y * image_height
        width = width * image_width
        height = height *image_height
        # compute the new coordinates
        x_min = center_x - width/2
        y_min = center_y - height/2
        x_max = x_min + width
        y_max = y_min + height
        # append converted bb
        bbs_list_converted.append([x_min, y_min, x_max, y_max])
    return bbs_list_converted
        

class AutonomousDrivingDataset(Dataset):
    def __init__(self, images_path, img_ext, image_width, image_height, labels_path, label_ext, classes, transform=None):
        self.files_list = [Path(img_name).stem for img_name in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, img_name))]
        self.images_path = images_path
        self.labels_path = labels_path
        self.img_ext = img_ext
        self.label_ext = label_ext
        self.image_width = image_width
        self.image_height = image_height
        self.classes = ['__background__'] + classes
        self.transform = transform

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        file_name = self.files_list[idx]
        img_path = os.path.join(self.images_path, file_name+self.img_ext)
        image = Image.open(img_path).convert("RGB")
        label_path = os.path.join(self.labels_path, file_name+self.label_ext)
        obj_labels, obj_bbs = parse_annotations_file(label_path)
        # convert bounding boxes coordinates from yolo format to the format used by RetinaNet
        obj_bbs = convert_bbs_from_yolo_to_retina(obj_bbs, self.image_width, self.image_height)
        
        # List with the class of each object/bounding box present in the image
        obj_labels = torch.as_tensor(obj_labels, dtype=torch.int64)
        # List of all the bounding boxes (bbs) present in the image
        obj_bbs = torch.as_tensor(obj_bbs, dtype=torch.float32)
        target = {"boxes":obj_bbs, "labels":obj_labels}
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, target
