import os
import cv2
import numpy as np

import torch
import random


def random_crop(image, boxes):
    height, width, _ = image.shape
    cw = random.randint(int(width * 0.75), width)
    ch = random.randint(int(height * 0.75), height)
    cx = random.randint(0, width - cw)
    cy = random.randint(0, height - ch)

    roi = image[cy:cy + ch, cx:cx + cw]
    roi_h, roi_w, _ = roi.shape

    output = []
    for box in boxes:
        index, category = box[0], box[1]
        bx = (box[2] * width - cx) / roi_w
        by = (box[3] * height - cy) / roi_h
        bw = box[4] * width / roi_w
        bh = box[5] * height / roi_h
        output.append([index, category, bx, by, bw, bh])

    return roi, np.array(output, dtype=float)


def random_narrow(image, boxes):
    height, width, _ = image.shape
    cw = random.randint(width, int(width * 1.25))
    ch = random.randint(height, int(height * 1.25))
    cx = random.randint(0, cw - width)
    cy = random.randint(0, ch - height)

    background = np.ones((ch, cw, 3), np.uint8) * 128
    background[cy:cy + height, cx:cx + width] = image

    output = []
    for box in boxes:
        index, category = box[0], box[1]
        bx = (box[2] * width + cx) / cw
        by = (box[3] * height + cy) / ch
        bw = box[4] * width / cw
        bh = box[5] * height / ch
        output.append([index, category, bx, by, bw, bh])

    return background, np.array(output, dtype=float)


def random_flip(image, boxes):
    """Horizontal flip with box coordinate mirroring."""
    image = cv2.flip(image, 1)
    output = []
    for box in boxes:
        index, category = box[0], box[1]
        bx = 1.0 - box[2]  # mirror cx
        by, bw, bh = box[3], box[4], box[5]
        output.append([index, category, bx, by, bw, bh])
    return image, np.array(output, dtype=float)


def color_jitter(image, brightness=0.3, contrast=0.3, saturation=0.3):
    """Apply random brightness, contrast, and saturation jitter."""
    img = image.astype(np.float32)

    # Brightness
    if random.random() < 0.5:
        delta = random.uniform(-brightness, brightness) * 255
        img = np.clip(img + delta, 0, 255)

    # Contrast
    if random.random() < 0.5:
        scale = random.uniform(1 - contrast, 1 + contrast)
        img = np.clip(img * scale, 0, 255)

    # Saturation (via HSV)
    if random.random() < 0.5:
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(1 - saturation, 1 + saturation), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    return img.astype(np.uint8)


def collate_fn(batch):
    img, label = zip(*batch)
    for i, l in enumerate(label):
        if l.shape[0] > 0:
            l[:, 0] = i
    return torch.stack(img), torch.cat(label, 0)


class TensorDataset:
    def __init__(self, path, img_width, img_height, aug=False):
        assert os.path.exists(path), "%s文件路径错误或不存在" % path

        self.aug = aug
        self.path = path
        self.data_list = []
        self.img_width = img_width
        self.img_height = img_height
        self.img_formats = ['bmp', 'jpg', 'jpeg', 'png']

        with open(self.path, 'r') as f:
            for line in f.readlines():
                data_path = line.strip()
                if not data_path:
                    continue
                if not os.path.exists(data_path):
                    raise FileNotFoundError("%s does not exist" % data_path)
                img_type = data_path.rsplit(".", 1)[-1].lower()
                if img_type not in self.img_formats:
                    raise ValueError("Unsupported image format: %s" % img_type)
                self.data_list.append(data_path)

    def __getitem__(self, index):
        img_path = self.data_list[index]
        label_path = img_path.rsplit(".", 1)[0] + ".txt"

        img = cv2.imread(img_path)

        if os.path.exists(label_path):
            label = []
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    l = line.strip().split(" ")
                    if len(l) == 5:
                        label.append([0, l[0], l[1], l[2], l[3], l[4]])
            label = np.array(label, dtype=np.float32)
            if label.shape[0]:
                assert label.shape[1] == 6, "> 5 label columns: %s" % label_path
        else:
            raise FileNotFoundError("%s does not exist" % label_path)

        if self.aug:
            # Geometric augmentations
            r = random.random()
            if r < 0.33:
                img, label = random_narrow(img, label)
            elif r < 0.66:
                img, label = random_crop(img, label)
            # else: no geometric transform

            # Horizontal flip (50%)
            if random.random() < 0.5:
                img, label = random_flip(img, label)

            # Color jitter (50%)
            if random.random() < 0.5:
                img = color_jitter(img)

        img = cv2.resize(img, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
        img = img.transpose(2, 0, 1)

        return torch.from_numpy(img), torch.from_numpy(label)

    def __len__(self):
        return len(self.data_list)
