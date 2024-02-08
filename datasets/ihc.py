import os

import cv2
import pandas as pd
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms as transforms_tv


from .build import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class IHC(torch.utils.data.Dataset):
    """IHC dataset."""

    def __init__(self, cfg, filePath, condition="normal", database="IHC", aug=False):
        self.cfg = cfg
        self.data_path = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, database)
        self.data_file = filePath
        self.condition = condition  # normal or pathology
        self.locations = cfg.CLASSIFIER.LOCATIONS
        self.annotations = dict(zip(self.locations, range(len(self.locations))))
        self.im_size = cfg.DATA.CROP_SIZE
        self.aug = aug
        self._get_img_info()


    def _get_img_info(self):
        data_file = pd.read_csv(self.data_file, header=0, index_col=0)
        self._data_info = []
        if 'locations' in data_file.columns:
            data_file = data_file[['URL', 'Pair Idx'] + self.locations]
            data_file[self.locations] = data_file[self.locations].astype(int)
            for item in data_file.itertuples(index=True):
                index, url, pairIdx = item[0 : 3]
                locations = item[3 : ]

                if pairIdx.split('-')[0] == "N":
                    condition = 0
                elif pairIdx.split('-')[0] == "P":
                    condition = 1
                else:
                    condition = -1
                pairIdx = int(pairIdx.split('-')[1])

                im_path = os.path.join(self.data_path, "normal" if condition == 0 else "pathology", url)

                self._data_info.append({"index": index, "im_path": im_path, "annotations": locations, "pairIdx": pairIdx, "condition": condition})
        else:
            data_file = data_file[['URL', 'Pair Idx']]
            for item in data_file.itertuples(index=True):
                index, url, pairIdx = item

                if pairIdx.split('-')[0] == "N":
                    condition = 0
                elif pairIdx.split('-')[0] == "P":
                    condition = 1
                else:
                    condition = -1
                pairIdx = int(pairIdx.split('-')[1])

                im_path = os.path.join(self.data_path, "normal" if condition == 0 else "pathology", url)

                self._data_info.append({"index": index, "im_path": im_path, "annotations": [], "pairIdx": pairIdx, "condition": condition})


    def __load__(self, index):
        im_path = self._data_info[index]["im_path"]
        # print(im_path)
        im = Image.open(im_path)
        # im = im.convert("YCbCr")  # YCbCr, where Y refers to the luminance component, Cb to the blue chrominance component, and Cr to the red chrominance component.
        im = im.convert("RGB")

        t = []
        if self.aug:
            t.append(transforms_tv.RandomRotation(degrees=(90)))
            t.append(transforms_tv.RandomHorizontalFlip(p=0.5))
            t.append(transforms_tv.RandomVerticalFlip(p=0.5))
        w, h = im.size
        if h < 1000 or w < 1000:
            t.append(transforms_tv.CenterCrop(0.9 * min(w, h)))
            if self.aug:
                t.append(transforms_tv.RandomCrop(min(w, h) * self.im_size // 3000))
            else:
                t.append(transforms_tv.CenterCrop(min(w, h) * self.im_size // 3000))
            t.append(transforms_tv.Resize([self.im_size, self.im_size]))
        else:
            if h < self.im_size or w < self.im_size:
                left_pad = top_pad = right_pad = bottom_pad = 0
                if h < self.im_size:
                    top_pad = (self.im_size + 1 - h) // 2
                    bottom_pad = self.im_size - h - top_pad
                if w < self.im_size:
                    left_pad = (self.im_size + 1 - w) // 2
                    right_pad = self.im_size - w - left_pad
                t.append(transforms_tv.Pad((left_pad, top_pad, right_pad, bottom_pad), padding_mode="reflect"))
            if h > self.im_size or w > self.im_size:
                t.append(transforms_tv.CenterCrop(max(0.9 * min(w, h), self.im_size)))
                if self.aug:
                    t.append(transforms_tv.RandomCrop(self.im_size))
                else:
                    t.append(transforms_tv.CenterCrop(self.im_size))
        t.append(transforms_tv.ToTensor())
        # t.append(transforms_tv.Normalize(self.cfg.DATA.MEAN, self.cfg.DATA.STD))
        transform = transforms_tv.Compose(t)
        im = transform(im)

        return im


    def __getitem__(self, index):
        im = self.__load__(index)
        idx = self._data_info[index]["index"]
        label = self._data_info[index]["annotations"]
        label = torch.FloatTensor(label)
        pairIdx = self._data_info[index]["pairIdx"]
        condition = self._data_info[index]["condition"]
        return idx, im, label, pairIdx, condition


    def __len__(self):
        return len(self._data_info)