import os
import cv2
import json
import torch
from torch.utils.data import Dataset
import albumentations as A
import numpy as np

def get_data(data_root, data_format):
    data = {
        os.path.relpath(os.path.join(root, fname), start=data_root)
        for root, _dirs, files in os.walk(data_root)
        for fname in files
        if os.path.splitext(fname)[1].lower() == data_format
        }
    return data

class XRayDataset(Dataset):
    def __init__(self,
                 images,
                 labels,
                 image_root,
                 label_root,
                 class_index,
                 class2ind,
                 is_train=True,
                 transform=None):
        self.filenames = np.array(images)
        self.labelnames = np.array(labels)

        self.image_root = image_root
        self.label_root = label_root

        self.class_index = class_index
        self.class2ind = class2ind

        self.is_train = is_train
        self.transforms = transform

        self.clahe_transform_A = A.CLAHE(clip_limit=1, tile_grid_size=(8, 8), p=1.0)
        self.clahe_transform_B = A.CLAHE(clip_limit=2, tile_grid_size=(8, 8), p=1.0)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)

        image = cv2.imread(image_path)
        # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # image_A = self.clahe_transform_A(image=image)["image"]
        # image_B = self.clahe_transform_B(image=image)["image"]
        # image = np.stack([image, image_A, image_B], axis=2)
        image = image / 255.

        label_name = self.labelnames[item]
        label_path = os.path.join(self.label_root, label_name)

        # (H, W, NC) 형태의 label 생성
        label_shape = tuple(image.shape[:2]) + (len(self.class_index), )
        label = np.zeros(label_shape, dtype=np.uint8)

        # label_xxx.json 읽어들이기
        with open(label_path, "r") as f:
            annotations = json.load(f)

        # self.class_index에 있는 인덱스에 한해서 마스크를 가져와 정답 라벨을 구성
        # 예시1. self.class_index = [0 ~ 28]  => 모든 뼈의 마스크 가져와 (H,W,29) 크기의 정답 라벨 구성
        # 예시2. self.class_index = [0 ~ 18]  => 손가락 뼈에 해당하는 마스크 가져와 (H,W,19) 크기의 정답 라벨 구성
        # 예시3. self.class_index = [19 ~ 26] => 손등 뼈에 해당하는 마스크 가져와 (H,W,8) 크기의 정답 라벨 구성
        # 예시4. self.class_index = [27 ~ 28] => 팔 뼈에 해당하는 마스크 가져와 (H,W,2) 크기의 정답 라벨 구성

        for ann in annotations:
            c = ann["label"]
            class_ind = self.class2ind[c]
            if class_ind not in self.class_index:
                continue
            points = np.array(ann["points"])

            # polygon 포맷을 dense한 mask 포맷으로 변환
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind-self.class_index[0]] = class_label

        # 이미지와 동일하게 상단, 좌단, 우단 80 픽셀 영 처리
        image_size = image.shape[0]
        if image_size == 2048:
            label[:80, :, :] = 0
            label[:, :80, :] = 0
            label[:, -80:, :] = 0

        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)

            image = result["image"]
            label = result["mask"] if self.is_train else label

        # (H,W,C) -> (C,H,W)
        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label
    
def get_transforms(config: dict, is_train=True):
    resize = config['augmentation']['train']['resize']

    transform_list = []
    transform_config = config['augmentation']['train']

    if transform_config.get('resize', False):
        transform_list.append(A.Resize(resize, resize))
        print("Reszie 적용")

    horizon_flip_p  = transform_config['horizontal_flip']
    scale_percent   = transform_config['scale']
    shift_percent   = transform_config['shift']
    rotate_degree   = transform_config['rotate']
    shear_degree    = transform_config['shear']
    crop_ratio      = transform_config['crop']
    elastic_p       = transform_config['elastic']

    if is_train:
        if transform_config.get('elastic', False):
            transform_list.append(A.ElasticTransform(alpha=300, sigma=10, p=elastic_p))
            print("Elastic transform 적용")

        if transform_config.get('horizontal_flip', False):
            transform_list.append(A.HorizontalFlip(p = horizon_flip_p))
            print("Horizontal flip 적용")

        if transform_config.get('crop', False):
            transform_list.append(A.RandomResizedCrop(size=(512,512),
                                                    scale=(0.5, 0.5),
                                                    ratio=(1,1),
                                                    p=0.8))
            print("Random Resized Crop 적용")

        if (transform_config.get('scale', False) != 0 or
            transform_config.get('shift', False) != 0 or
            transform_config.get('shear', False) != 0 or
            transform_config.get('rotate', False) != 0):
            transform_list.append(A.Affine(scale=[1-scale_percent, 1+scale_percent],
                                           translate_percent=[-shift_percent, shift_percent],
                                           rotate=[-rotate_degree, rotate_degree],
                                           shear=[-shear_degree, shear_degree],
                                           p=0.8))
            print("Affine 적용")
        
    if transform_config.get('normalize', False):
        transform_list.append(A.Normalize(mean=(0.485, 0.456, 0.406),
                                          std=(0.229, 0.224, 0.225),
                                          max_pixel_value=1))
        print("Normalize 적용")

    return A.Compose(transform_list)

class OutputDataset(Dataset):
    def __init__(self, output, label, output_root, label_root):
        self.output = output
        self.label = label

        self.output_root = output_root
        self.label_root = label_root

        assert len(self.output) == len(self.label)

    def __len__(self):
        return len(self.output)
    
    def __getitem__(self, idx):
        output_name = self.output[idx]
        label_name = self.label[idx]

        output_path = os.path.join(self.output_root, output_name)
        label_path = os.path.join(self.label_root, label_name)

        output = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE) // 255
        label  = cv2.imread(label_path,  cv2.IMREAD_GRAYSCALE) // 255

        return output, label