# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T
from .random_erasing import RandomErasing

# class Transforms():

#     # def __init__(self, params):
#     #     self.params = params

#     def build_transforms(self, is_train=True):
#         normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.485, 0.456, 0.406])
#         if is_train:
#             transform = T.Compose([
#                 T.Resize([256, 128]),
#                 # T.Resize((224, 224)),
#                 T.RandomHorizontalFlip(p=0.5),
#                 T.Pad(10),
#                 T.RandomCrop(0.5),
#                 T.ToTensor(),
#                 normalize_transform,
#                 RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
#             ])
#         else:
#             transform = T.Compose([
#                 T.Resize([256, 128]),
#                 T.ToTensor(),
#                 normalize_transform
#             ])

#         return transform
    
def build_transforms(is_train=True):
    normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.485, 0.456, 0.406])
    inp_size = [256, 128]
    if is_train:
        transform = T.Compose([
            T.Resize(inp_size),
            # T.Resize((224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop(inp_size),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
        ])
    else:
        transform = T.Compose([
            T.Resize(inp_size),
            # T.Resize((224, 224)),
            T.ToTensor(),
            normalize_transform
        ])

    return transform