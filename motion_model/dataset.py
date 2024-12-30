import os
import torch.utils.data as data
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from scene.deformable_field import positional_encoding
import json
import numpy as np

class GCNBaseDataset(data.Dataset, ABC):
    def __init__(self, gaussians, time_freq, iteration, model_path, source_path, max_time=0.8, input_size=20, output_size=5, split="train"):
        super().__init__()
        self.gaussians = gaussians
        self.time_freq = time_freq
        self.iteration = iteration
        self.split = split
        self.max_time = max_time
        assert self.max_time < 1.0
        self.input_size = input_size
        self.output_size = output_size
        self.model_path = model_path
        self.source_path = source_path

        self.train_data = []
        self.test_data = []
        self.val_data = []

        self.process_data()

    def load_hyper_times(self):
        datadir = self.source_path
        with open(f'{datadir}/metadata.json', 'r') as f:
            meta_json = json.load(f)
        with open(f'{datadir}/dataset.json', 'r') as f:
            dataset_json = json.load(f)
        self.all_img = dataset_json['ids']
        self.val_id = dataset_json['val_ids']
        self.all_time = [meta_json[i]['warp_id'] for i in self.all_img]

        max_time = max(self.all_time)
        if self.max_time < 1.0:
            self.all_time, self.i_train, self.i_test = [], [], []
            for idx, i in enumerate(self.all_img):
                time = meta_json[i]['warp_id']/max_time
                self.all_time += [time]
                if len(self.val_id) == 0:
                    if idx % 4 ==0 and time < self.max_time:
                        self.i_train += [idx]
                    if (idx - 2) % 4 ==0 and time >= self.max_time:
                        self.i_test += [idx]
                else:
                    self.train_id = dataset_json['train_ids']
                    if i in self.val_id and time >= self.max_time:
                        self.i_test.append(idx)
                    if i in self.train_id and time < self.max_time:
                        self.i_train.append(idx)
            self.i_test = np.array(self.i_test)
            self.i_train = np.array(self.i_train)
            np_all_time = np.array(self.all_time)
            test_time = np_all_time[self.i_test]
            train_time = np_all_time[self.i_train]
            assert test_time.max() >= self.max_time
            assert train_time.min() < self.max_time
            print("train:", self.i_train)
            print("test:", self.i_test)
        else:
            self.all_time = [meta_json[i]['warp_id']/max_time for i in self.all_img]

        self.test_times = [self.all_time[idx] for idx in self.i_test]
        self.train_times = [self.all_time[idx] for idx in self.i_train]
        if self.split == "test":
            print(f"Check!!!!!There are {len(self.test_times)} test data!!!!")
        

    def load_dnerf_times(self):
        with open(os.path.join(self.source_path, "transforms_train.json"), "r") as train_file:
            data = json.load(train_file)["frames"]
            self.train_times, self.test_times = [], []
            for i in range(len(data)):
                if float(data[i]["time"]) < self.max_time:
                    self.train_times.append(float(data[i]["time"]))
                else:
                    self.test_times.append(float(data[i]["time"]))
        train_file.close()

    def load_times(self):
        if "d-nerf" in self.model_path or "white" in self.model_path:
            print("Find transforms_train.json file, assume it D-NeRF Dataset!")
            self.load_dnerf_times()
        else:
            print("Find metadata.json file, assume it HyperNeRF Dataset!")
            self.load_hyper_times()
    
    @abstractmethod
    def get_lens(self):
        pass

    @abstractmethod
    def prepare_item(self):
        pass

    def process_data(self):
        self.load_times()
        self.generate_data()
        self.get_lens()
        self.prepare_item()

    @property
    def nodes_num(self):
        return self.gaussians.super_gaussians.shape[0]
    
    def generate_data(self):
        kpts_xyz, kpts_r = [], []
        self.kpts_xyz_motion_train, self.kpts_r_motion_train = [], []
        self.kpts_xyz_motion_test, self.kpts_r_motion_test = [], []
        for i in range(len(self.train_times)):
            time_ = torch.tensor([self.train_times[i]], dtype=torch.float32).cuda()
            _, _, _, _ = self.gaussians(time_, self.iteration)
            kpts_xyz += [self.gaussians.get_superGaussians + self.gaussians.kpts_xyz_motion]
            kpts_r += [self.gaussians.kpts_rotation_motion]
            self.kpts_xyz_motion_train += [self.gaussians.kpts_xyz_motion]
            self.kpts_r_motion_train += [self.gaussians.kpts_rotation_motion]
        self.kpts_xyz_train = torch.stack(kpts_xyz, dim=0)
        self.kpts_r_train = torch.stack(kpts_r, dim=0)
        kpts_xyz, kpts_r = [], []
        for i in range(len(self.test_times)):
            time_ = torch.tensor([self.test_times[i]], dtype=torch.float32).cuda()
            _, _, _, _ = self.gaussians(time_, self.iteration)
            kpts_xyz += [self.gaussians.get_superGaussians + self.gaussians.kpts_xyz_motion]
            kpts_r += [self.gaussians.kpts_rotation_motion]
            self.kpts_xyz_motion_test += [self.gaussians.kpts_xyz_motion]
            self.kpts_r_motion_test += [self.gaussians.kpts_rotation_motion]
        self.kpts_xyz_test = torch.stack(kpts_xyz, dim=0)
        self.kpts_r_test = torch.stack(kpts_r, dim=0)

    def __len__(self):
        if self.split == "train":
            # return self.train_lens
            return len(self.train_data)
        elif self.split == "test":
            # return self.test_lens
            return len(self.test_data)
        else:
            # return self.val_lens
            return len(self.val_data)
        
class GCN3DDataset(GCNBaseDataset):
    def get_lens(self):
        self.train_lens = len(self.kpts_xyz_train) - self.input_size - self.output_size
        self.test_lens = len(self.kpts_xyz_test)
        self.val_lens = 2
    
    def prepare_item(self):
        
        if self.split == "train":
            for i in trange(self.train_lens, desc="loading training data"):
                xyz_input = self.kpts_xyz_train[i:i+self.input_size]
                xyz_gt = self.kpts_xyz_train[i+self.input_size:i+self.input_size+self.output_size]
                time_interval = self.train_times[i+self.input_size] - self.train_times[i+self.input_size - 1]
                rotation_input = self.kpts_r_train[i:i+self.input_size]
                rotation_gt = self.kpts_r_train[i+self.input_size:i+self.input_size+self.output_size]
                self.train_data.append({"xyz_inputs": xyz_input, "xyz_gt": xyz_gt, 
                                        "rotation_inputs": rotation_input, "rotation_gt": rotation_gt, "time": time_interval})
        elif self.split == "test":
            concat_kpts_xyz_test = torch.cat([self.kpts_xyz_train[-self.input_size:], self.kpts_xyz_test], dim=0)
            concat_kpts_r_test = torch.cat([self.kpts_r_train[-self.input_size:], self.kpts_r_test], dim=0)
            concat_times = self.train_times[-self.input_size:] + self.test_times
            for i in trange(0, self.test_lens, self.output_size, desc="loading testing data"):
                xyz_input = concat_kpts_xyz_test[i:i+self.input_size]
                xyz_gt = concat_kpts_xyz_test[i+self.input_size:i+self.input_size+self.output_size]
                time_interval = concat_times[i+self.input_size] - concat_times[i+self.input_size - 1]
                rotation_input = concat_kpts_r_test[i:i+self.input_size]
                rotation_gt = concat_kpts_r_test[i+self.input_size:i+self.input_size+self.output_size]
                self.test_data.append({"xyz_inputs": xyz_input, "xyz_gt": xyz_gt,
                                       "rotation_inputs": rotation_input, "rotation_gt": rotation_gt, "time": time_interval})
        else:
            idx = torch.randint(0, self.train_lens, [1]).item()
            self.val_data.append(self.train_data[idx])
            last_test_xyz = self.kpts_xyz_test[-self.input_size:]
            last_test_r = self.kpts_r_test[-self.input_size:]
            self.val_data.append({"xyz_inputs": last_test_xyz, "xyz_gt": torch.zeros([self.output_size, *last_test_xyz.shape[1:]]).to(last_test_xyz),
                                  "rotation_inputs": last_test_r, "rotation_gt": torch.zeros([self.output_size, *last_test_r.shape[1:]]).to(last_test_r)})

    def __getitem__(self, index):
        if self.split == "train":
            return self.train_data[index]
        elif self.split == "test":
            return self.test_data[index]
        else:
            return self.val_data[index]
