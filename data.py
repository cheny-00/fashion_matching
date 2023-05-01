import os
import json
import torch
import random
import numpy as np

from PIL import Image
from collections import defaultdict

from utils import get_transform

class BaseDataset:
    
    def __init__(self, files_dict, device='cpu', transform=None) -> None:
        self.data = files_dict
        self.device = device
        self.transform = transform
        
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError
    
    def prepare_img(self, img_path):
        sample = load_image(img_path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    @staticmethod
    def save_dataset_record(records, save_path):
        with open(save_path, "w") as f:
            json.dump(records, f)

    @staticmethod
    def load_dataset_record(save_path):
        with open(save_path, 'r') as f:
            records = json.load(f)
        return records
        

class DFSiameseDataset(BaseDataset):
    
    def __init__(self, *args, use_record=False, load_record_path=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.image_pair = list()
        self.labels = list(self.data.keys())
        self.n_discard = 0

        self.record_pairs_id = list()
        self.image_pair = list()
        if use_record:
            self.record_pairs_id = self.load_dataset_record(load_record_path)
        
        # self.set_dataset_windows_size(3)
        
        self.sample_dataset(use_record)
    
    def sample_dataset(self, use_record):
        # if self.dataset_windows_size < len(self.image_pair):
        #     self.image_pair = list()
        # self.image_pair = list()
        if use_record:
            self._build_record_image_pairs()
        else:
            self._build_random_image_pairs()
        print(f"discard {self.n_discard}")
        self.n_discard = 0

    def __len__(self, ):
        return len(self.image_pair)

    def set_dataset_windows_size(self, n_windows):
        self.dataset_windows_size = len(self.labels) * n_windows
    
    def _build_record_image_pairs(self,):
        for r in self.record_pairs_id:
            x_class, x_idx, y_class, y_idx, target = \
            r['x_class'], r['x_idx'], r['y_class'], r['y_idx'], r['target']
            x = self.data[x_class][x_idx]
            y = self.data[y_class][y_idx]
            self.image_pair.append((x, y, target))

    def _build_random_image_pairs(self):
        for pair_id in self.labels:
            if len(self.data[pair_id]) == 1:
                self.n_discard += 1
                continue
            x_idx = random.choice(list(range(len(self.data[pair_id]))))
            x = self.data[pair_id][x_idx]
            is_positive = random.choice([True, False])

            if is_positive:
                y_class = pair_id
                y_idx = random.choice([i for i in range(len(self.data[pair_id])) if i != x_idx])
                y = self.data[pair_id][y_idx]
                target = 1.0
            else:
                y_class = random.choice([_cls for _cls in self.data.keys() if _cls != pair_id])
                y_idx = random.choice(range(len(self.data[y_class])))
                y = self.data[y_class][y_idx]
                # y = random.choice(self.data[random.choice(y_class)])
                target = 0.0
            
            self.record_pairs_id.append(
                {
                    "x_class" : pair_id,
                    "x_idx": x_idx,
                    "y_class": y_class,
                    "y_idx": y_idx,
                    "target": target
                }
            )

            self.image_pair.append((x, y, target))

    def __getitem__(self, idx):
        
        features = dict()
        x, y, target = self.image_pair[idx]
        features['x'] = self.transform(load_image(x)).to(self.device)
        features['y'] = self.transform(load_image(y)).to(self.device)
        features['target'] =  torch.tensor(target, dtype=torch.float).to(self.device)
        return features


class DFTpripletTrainDataset(BaseDataset):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.image_pair = list()
        self.labels = list(self.data.keys())
        self.n_discard = 0
        self.sample_dataset()
        
    
    def __getitem__(self, idx):
        
        features = dict()
        anchor_img, positive_img, negative_img = self.image_pair[idx]
        features['anchor'] = self.transform(load_image(anchor_img)).to(self.device)
        features['positive'] = self.transform(load_image(positive_img)).to(self.device)
        features['negative'] = self.transform(load_image(negative_img)).to(self.device)
        return features
        
        
    def sample_dataset(self, ):
        self._build_image_pairs()
        # print(f"discard {self.n_discard}")
        self.n_discard = 0

    def __len__(self, ):
        return len(self.image_pair)
    
    def _build_image_pairs(self,):
        for pair_id in self.labels:
            if len(self.data[pair_id]) == 1:
                # print(f"{pair_id} was discarded")
                self.n_discard += 1
                continue
            anchor_class = pair_id
            negative_class = random.choice([_cls for _cls in self.data.keys() if _cls != anchor_class])

            anchor_img = random.choice(self.data[anchor_class])
            positive_img = random.choice([img for img in self.data[anchor_class] if img != anchor_img])
            negative_img = random.choice(self.data[negative_class])

            self.image_pair.append((anchor_img, positive_img, negative_img))




class TripletEmbedDataset(BaseDataset):
    
    def __init__(self, files_dict, n_instances, do_resample, device='cpu', transform=None , *args, **kwargs) -> None:
        super().__init__(files_dict, device, transform, *args, **kwargs)
        
        self._form_dataset()
        self.n_instances = n_instances
        self.do_resample = do_resample

        self.n_classes = len(self.data)
        
        self.labels = list(self.data.keys())
        self.labels2indices = {label: idx for idx, label in enumerate(self.labels)}
        
    
    def _form_dataset(self, ):
        new_data = defaultdict(list)
        for pid, samples in self.data.items():
            if not len(samples) > 1:
                continue
            samples = [(_s, _i) for _i, _s in enumerate(samples)]
            new_data[pid] = samples
        self.data = new_data
        
        
    def _build_single_sample(self, pid):
        label = self.labels[pid]
        samples = self.data[label][:]
        n_sample = len(samples)
        # if not  n_sample > 1:
        #     return 
        if n_sample < self.n_instances:
            choice_size = n_sample
            need_pad = True
        else:
            choice_size = self.n_instances
            need_pad = False
            
        random.shuffle(self.data[pid])
        
        new_samples = defaultdict(list)
        
        for _ in range(choice_size):
            img_path, idx = self.data[label].pop(0)
            img = self.prepare_img(img_path)
            new_samples['x'].append(img)
            new_samples['label'].append(pid)
            new_samples['idx'].append(idx)
            new_samples['is_real'].append(True)
        
        if need_pad:
            
            n_missing = self.n_instances - n_sample
            if self.do_resample:
                resampled = np.random.choice(
                    range(len(samples)), size=n_missing, replace=True
                )
                for i in resampled:
                    img_path, idx = samples[i]
                    img = self.prepare_img(img_path)
                    new_samples['x'].append(img)
                    new_samples['label'].append(pid)
                    new_samples['idx'].append(idx)
                    new_samples['is_real'].append(True)
            else:
                img_mock = torch.zeros_like(img)
                for _ in range(n_missing):
                    new_samples['x'].append(img_mock)
                    new_samples['label'].append(pid)
                    new_samples['idx'].append(idx)
                    new_samples['is_real'].append(False)
        return new_samples
    
    def __len__(self):
        return len(self.labels)
    
    
    def get_samples(self, ):
        batch_samples = list()
        for i in range(self.__len__):
            batch_samples.append(self._build_single_sample(i))
        self.batch_samples = batch_samples
    def __getitem__(self, pid):
        return self._build_single_sample(pid)

        

     
    
def per_class_instances(data_dict):
    
    num_table = defaultdict(int)
    print(len(data_dict))
    for pid, samples in data_dict.items():
        s_size = len(samples)
        num_table[s_size] += 1
        if s_size == 8:
            print(pid)
    print(num_table)
        
        
        

    
def read_single_pic(img_path, transform):
    return transform(load_image(img_path))



def get_label(filename):
    label_list = list()
    for char in filename:
        if char.isnumeric():
            label_list.append(char)
    return "".join(label_list[:-1])

def get_filenames(target_dir, build_labels_fuc):
    files_dict = defaultdict(list)
    fnames = os.listdir(target_dir)
    for filename in fnames:
        label = build_labels_fuc(filename)
        files_dict[label].append(os.path.join(target_dir, filename))
    return files_dict

# def load_images(paths):
#     for _path in paths:
#         yield load_image(_path)
    
def load_image(image_path):
    with open(image_path, "rb") as f:
        img = Image.open(f).convert('RGB')
    return img
       
def batch_convert_fn(batch): # add random shuffler => ranndom anchor
    new_features = defaultdict(list)
    for data in batch:
        for k, v in data.items():
            new_features[k].extend(v)
    new_features['x'] = torch.stack(new_features['x'])
    new_features['label'] = torch.tensor(new_features['label'])
    new_features['idx'] = torch.tensor(new_features['idx'])
    new_features['is_real'] = torch.tensor(new_features['is_real'])
    # for i in range(128):
    #     m = 128 % 4
    #     n = m // 4
    #     f = new_features['x'] ==  batch[n]['x'][m]
    #     print(f)
    return new_features



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    train_data_dir = "deepfashion_train_test_256/train_test_256/train"
    test_data_dir = "deepfashion_train_test_256/train_test_256/test"
    
    train_files_dict = get_filenames(train_data_dir, build_labels_fuc=get_label)
    # train_dataset = DFTpripletTrainDataset(train_files_dict)
    # train_dataset = DFSiameseDataset(train_files_dict)
    # print(len(os.listdir(train_data_dir)), len(train_dataset))
    # test_files_dict = get_filenames(test_data_dir, build_labels_fuc=get_label)
    # test_dataset = DFSiameseDataset(test_files_dict, 0.5)
    # print(len(test_dataset))
    
    # per_class_instances(train_files_dict)
    train_dataset = TripletEmbedDataset(train_files_dict, n_instances=4, do_resample=True, transform=get_transform())
    print(len(train_dataset))
    
    train_iter = DataLoader(train_dataset, batch_size=128 // 4, shuffle=False, drop_last=True,
                            collate_fn=batch_convert_fn)
    for t in train_iter:
        print(t['label'])
    