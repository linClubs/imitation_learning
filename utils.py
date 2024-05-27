import numpy as np
import torch
import os
import h5py
import pickle
import fnmatch
import cv2
from time import time
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms

import IPython
e = IPython.embed

def flatten_list(l):
    return [item for sublist in l for item in sublist]

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path_list, camera_names, norm_stats, episode_ids, episode_len, chunk_size, policy_class):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_path_list = dataset_path_list
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.episode_len = episode_len
        self.chunk_size = chunk_size
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(episode_len)
        self.policy_class = policy_class
        if self.policy_class == 'Diffusion':
            self.augment_images = True
        else:
            self.augment_images = False
        self.transformations = None
        self.__getitem__(0) # initialize self.is_sim and self.transformations
        self.is_sim = False

    # def __len__(self):
    #     return sum(self.episode_len)

    # index = 5, episode_index = 6, 
    def _locate_transition(self, index):  # 假如是5
        assert index < self.cumulative_len[-1]  # [0, 500]  index必须小于max_timestamp
        # 如果500查找索引的位置  返回第一个大于 index 的元素的索引位置
        # index从0开始遍历, 
        # index = 2
        # print(self.episode_len)    # [500, 500, 500,...,500]
        # print(self.cumulative_len) # [500, 1000, 1500,...,24500]
        # print(self.cumulative_len[-1])
        episode_index = np.argmax(self.cumulative_len > index) # argmax returns first True index
        # print(index, episode_index)
        # exit(-1)
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts

    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index)
       
        dataset_path = self.dataset_path_list[episode_id]
        try:
            # print(dataset_path)
            with h5py.File(dataset_path, 'r') as root:
                try: # some legacy data does not have this attribute
                    is_sim = root.attrs['sim']
                except:
                    is_sim = False
                compressed = root.attrs.get('compress', False)
                if '/base_action' in root:
                    base_action = root['/base_action'][()]
                    base_action = preprocess_base_action(base_action)
                    # 加上底盘速度
                    action = np.concatenate([root['/action'][()], base_action], axis=-1)
                else:  
                    action = root['/action'][()]
                    dummy_base_action = np.zeros([action.shape[0], 2])
                    action = np.concatenate([action, dummy_base_action], axis=-1)
                # action.shape = [500, 16], 没有底盘速度就补2个维度 线速度和角速度
                original_action_shape = action.shape  # [500, 16]
                episode_len = original_action_shape[0]
                # get observation at start_ts only
                qpos = root['/observations/qpos'][start_ts]
                qvel = root['/observations/qvel'][start_ts]
                
                image_dict = dict()
                for cam_name in self.camera_names:
                    image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
                
                if compressed:
                    for cam_name in image_dict.keys():
                        decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
                        image_dict[cam_name] = np.array(decompressed_image)
                
                # get all actions after and including start_ts
                if is_sim:
                    action = action[start_ts:]
                    action_len = episode_len - start_ts
                else:
                    action = action[max(0, start_ts - 1):] # hack, to make timesteps more aligned
                    action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

            # self.is_sim = is_sim  
            # padded_action = [500，16] 从ts_start时刻开始的action都取值
            padded_action = np.zeros((self.max_episode_len, original_action_shape[1]), dtype=np.float32)
            padded_action[:action_len] = action  # 将序列的起始值改到index=0处

            is_pad = np.zeros(self.max_episode_len)  # 500个0  
            # 序列从index=0开始到action_len结束, index=action_len后全部为1需要padding
            is_pad[action_len:] = 1 # 500-ts_start后全部为1  action_len = episode_len - start_ts

            # 只取chunk_size个数据
            padded_action = padded_action[:self.chunk_size]  
            is_pad = is_pad[:self.chunk_size]        # 0-32为0

            # new axis for different cameras 处理图像数据, 按第一个维度拼接 3，3，640，480
            all_cam_images = []
            for cam_name in self.camera_names:
                all_cam_images.append(image_dict[cam_name])
            all_cam_images = np.stack(all_cam_images, axis=0)

            # construct observations
            image_data = torch.from_numpy(all_cam_images)
            qpos_data = torch.from_numpy(qpos).float()
            action_data = torch.from_numpy(padded_action).float()
            is_pad = torch.from_numpy(is_pad).bool()

            # channel last
            image_data = torch.einsum('k h w c -> k c h w', image_data)

            # augmentation
            if self.transformations is None:
                print('Initializing transformations')
                original_size = image_data.shape[2:]
                ratio = 0.95
                self.transformations = [
                    # 裁剪的目标尺寸
                    transforms.RandomCrop(size=[int(original_size[0] * ratio), int(original_size[1] * ratio)]),
                    # 调整大小
                    transforms.Resize(original_size, antialias=True),
                    # 随机旋转
                    transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                    # 允许在亮度、对比度、饱和度和色相等方面对图像进行随机变换
                    transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5) #, hue=0.08)
                ]

            if self.augment_images:
                for transform in self.transformations:
                    image_data = transform(image_data)

            # normalize image and change dtype to float
            image_data = image_data / 255.0

            if self.policy_class == 'Diffusion':
                # normalize to [-1, 1]
                action_data = ((action_data - self.norm_stats["action_min"]) / (self.norm_stats["action_max"] - self.norm_stats["action_min"])) * 2 - 1
            else:
                # normalize to mean 0 std 1
                action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

            qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        except:
            print(f'Error loading {dataset_path} in __getitem__')
            quit()

        # print(image_data.dtype, qpos_data.dtype, action_data.dtype, is_pad.dtype)
        # print(image_data.shape, qpos_data.shape, action_data.shape, is_pad.shape)
        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_path_list):
    all_qpos_data = []
    all_action_data = []
    all_episode_len = []

    for dataset_path in dataset_path_list:
        try:
            with h5py.File(dataset_path, 'r') as root:
                qpos = root['/observations/qpos'][()]
                qvel = root['/observations/qvel'][()]
                if '/base_action' in root:
                    base_action = root['/base_action'][()]
                    base_action = preprocess_base_action(base_action)
                    action = np.concatenate([root['/action'][()], base_action], axis=-1)
                else:
                    action = root['/action'][()]
                    dummy_base_action = np.zeros([action.shape[0], 2])
                    action = np.concatenate([action, dummy_base_action], axis=-1)
        except Exception as e:
            print(f'Error loading {dataset_path} in get_norm_stats')
            print(e)
            quit()
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
        all_episode_len.append(len(qpos))
    
    # 读入所有数据，并按第一个维度上组合  默认50组, 500timestamp, qpos是14, action是16
    all_qpos_data = torch.cat(all_qpos_data, dim=0)  # [25000，14]
    all_action_data = torch.cat(all_action_data, dim=0) # [25000，16]

    # normalize action data
    action_mean = all_action_data.mean(dim=[0]).float()  # 求平均值 16维度
    action_std = all_action_data.std(dim=[0]).float()    # 求方差 16维度
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping剪掉，截断 最小值0.01和最大值np.inf正无穷大

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0]).float()
    qpos_std = all_qpos_data.std(dim=[0]).float()
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()
    
    # print(action_max, action_min)
    # print(len(action_std), len(qpos_std))
   
    eps = 0.0001  # 最小值和最大值都扩大一个小量 并转成numpy
    stats = {"action_mean": action_mean.numpy(), "action_std": action_std.numpy(),
             "action_min": action_min.numpy() - eps,"action_max": action_max.numpy() + eps,
             "qpos_mean": qpos_mean.numpy(), "qpos_std": qpos_std.numpy(),
             "example_qpos": qpos}

    return stats, all_episode_len

def find_all_hdf5(dataset_dir, skip_mirrored_data):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename: continue
            if skip_mirrored_data and 'mirror' in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    print(f'Found {len(hdf5_files)} hdf5 files')
    return hdf5_files

def BatchSampler(batch_size, episode_len_l, sample_weights):
    sample_probs = np.array(sample_weights) / np.sum(sample_weights) if sample_weights is not None else None
    sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in episode_len_l])
    while True:
        batch = []
        # 采集32次
        for _ in range(batch_size):
            episode_idx = np.random.choice(len(episode_len_l), p=sample_probs) #  从0-32随机选一个值
            step_idx = np.random.randint(sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1])
            batch.append(step_idx)
        yield batch
    # 使用 yield 关键字的函数不是一次性返回所有结果，
    # 而是每次调用时返回一个结果，并保持函数的执行状态，以便下一次调用继续从上次停止的位置开始。


def load_data(dataset_dir_l, name_filter, camera_names, batch_size_train, batch_size_val, chunk_size, skip_mirrored_data=False, load_pretrain=False, policy_class=None, stats_dir_l=None, sample_weights=None, train_ratio=0.9):
    if type(dataset_dir_l) == str:
        dataset_dir_l = [dataset_dir_l]
    # 读取数据集文件路径
    dataset_path_list_list = [find_all_hdf5(dataset_dir, skip_mirrored_data) for dataset_dir in dataset_dir_l]
    
    # 数据组数 eg:50组
    num_episodes_0 = len(dataset_path_list_list[0])
    
    # dataset_path_list = flatten_list(dataset_path_list_list)
    dataset_path_list = [item for sublist in dataset_path_list_list for item in sublist]

    # 过滤摄像头, 不使用摄像头就屏蔽了
    dataset_path_list = [n for n in dataset_path_list if name_filter(n)]

    num_episodes_l = [len(dataset_path_list) for dataset_path_list in dataset_path_list_list]
    num_episodes_cumsum = np.cumsum(num_episodes_l)

    # obtain train test split on dataset_dir_l[0] 打乱顺序，获取训练集和测试集划分
    shuffled_episode_ids_0 = np.random.permutation(num_episodes_0)  # 随机打乱[0,1,2...]
    train_episode_ids_0 = shuffled_episode_ids_0[:int(train_ratio * num_episodes_0)] # 取0.9
    val_episode_ids_0 = shuffled_episode_ids_0[int(train_ratio * num_episodes_0):]   # 取剩下的0.1
    
    train_episode_ids_l = [train_episode_ids_0] + [np.arange(num_episodes) + num_episodes_cumsum[idx] for idx, num_episodes in enumerate(num_episodes_l[1:])]
    val_episode_ids_l = [val_episode_ids_0]
    
    train_episode_ids = np.concatenate(train_episode_ids_l)  # 成了一个list了
    val_episode_ids = np.concatenate(val_episode_ids_l)
    
    print(f'\033[32m\nData from: {dataset_dir_l}\n- Train on {[len(x) for x in train_episode_ids_l]} episodes\n- Test on {[len(x) for x in val_episode_ids_l]} episodes\n\033[0m')

    
    if(len(train_episode_ids) == 0 or len(val_episode_ids) == 0):
        print("\033[31m\nDataset is not exists, please check the path of dataset.\033[0m")
        exit(-1)
    
    # 获取统计学值 均值和方差 只有action和joint
    # obtain normalization stats for qpos and action 加载预训练里面的统计值
    # if load_pretrain:
    #     with open(os.path.join('/home/zfu/interbotix_ws/src/act/ckpts/pretrain_all', 'dataset_stats.pkl'), 'rb') as f:
    #         norm_stats = pickle.load(f)
    #     print('Loaded pretrain dataset stats')

    _, all_episode_len = get_norm_stats(dataset_path_list)

    # 每组(回合)数据长度为500, 默认值
    train_episode_len_l = [[all_episode_len[i] for i in train_episode_ids] for train_episode_ids in train_episode_ids_l]
    val_episode_len_l = [[all_episode_len[i] for i in val_episode_ids] for val_episode_ids in val_episode_ids_l]
    
    train_episode_len = flatten_list(train_episode_len_l)
    val_episode_len = flatten_list(val_episode_len_l)
    
    # 正态分布参数的文件路径
    if stats_dir_l is None:
        stats_dir_l = dataset_dir_l
    elif type(stats_dir_l) == str:
        stats_dir_l = [stats_dir_l]
    
    # 获得正态分布参数 返回了action-16维的均值(截断), 方差,max+小量, min-小量; qpos-14维的均值(截断), 方差，一个qpos的样本数据
    norm_stats, _ = get_norm_stats(flatten_list([find_all_hdf5(stats_dir, skip_mirrored_data) for stats_dir in stats_dir_l]))
    print(f'Norm stats from: {stats_dir_l}')
    # print(norm_stats)
    
    # 自定义的批次采样器，每批次5个样本 batch_sampler=batch_sampler_train
    batch_sampler_train = BatchSampler(batch_size_train, train_episode_len_l, sample_weights)
    batch_sampler_val = BatchSampler(batch_size_val, val_episode_len_l, None)

    # print(f'train_episode_len: {train_episode_len}, val_episode_len: {val_episode_len}, train_episode_ids: {train_episode_ids}, val_episode_ids: {val_episode_ids}')

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, train_episode_ids, train_episode_len, chunk_size, policy_class)
    val_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, val_episode_ids, val_episode_len, chunk_size, policy_class)
    
    num_workers = 8
    num_workers = 8
    # train_num_workers = (8 if os.getlogin() == 'zfu' else 16) if train_dataset.augment_images else 2
    # val_num_workers = 8 if train_dataset.augment_images else 2
    # print(f'Augment images: {train_dataset.augment_images}, train_num_workers: {train_num_workers}, val_num_workers: {val_num_workers}')
    
    # 创建dataLoader
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, pin_memory=True, num_workers=num_workers, prefetch_factor=2)
    val_dataloader = DataLoader(val_dataset, batch_sampler=batch_sampler_val, pin_memory=True, num_workers=num_workers, prefetch_factor=2)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

def calibrate_linear_vel(base_action, c=None):
    if c is None:
        c = 0.0 # 0.19
    v = base_action[..., 0]
    w = base_action[..., 1]
    base_action = base_action.copy()
    base_action[..., 0] = v - c * w
    return base_action

def smooth_base_action(base_action):
    return np.stack([
        np.convolve(base_action[:, i], np.ones(5)/5, mode='same') for i in range(base_action.shape[1])
    ], axis=-1).astype(np.float32)

def preprocess_base_action(base_action):
    # base_action = calibrate_linear_vel(base_action)
    base_action = smooth_base_action(base_action)

    return base_action

def postprocess_base_action(base_action):
    linear_vel, angular_vel = base_action
    linear_vel *= 1.0
    angular_vel *= 1.0
    # angular_vel = 0
    # if np.abs(linear_vel) < 0.05:
    #     linear_vel = 0
    return np.array([linear_vel, angular_vel])

### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
