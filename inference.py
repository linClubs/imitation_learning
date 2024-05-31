import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import repeat
from tqdm import tqdm
from einops import rearrange
import time
from torchvision import transforms
import collections

from utils import load_data # data functions
from utils import compute_dict_mean, set_seed, detach_dict
from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy

from detr.models.latent_model import Latent_Model_Transformer

import IPython
e = IPython.embed

from imitate_episodes import make_policy, make_optimizer, forward_pass, repeater, plot_history


def get_image(observation, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(observation['images'][cam_name], 'h w c -> c h w')
    
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


# 1 读取配置文件, 创建dataloader
def main(args):
    set_seed(1)

    # TASK_CONFIGS配置
    task_name = args['task_name']
   
    

    # command line parameters
    ckpt_dir = args['ckpt_dir']
    ckpt_name = args['ckpt_name']
    stats_name = args['stats_name']

    policy_class = args['policy_class']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']
    eval_every = args['eval_every']
    validate_every = args['validate_every']
    save_every = args['save_every']
    resume_ckpt_path = args['resume_ckpt_path']

    # get task parameters
    # num_episodes = task_config['num_episodes']
    episode_len = args['episode_len']
    camera_names = args['camera_names']
    stats_dir = args.get('stats_dir', None)          
    sample_weights = args.get('sample_weights', None)
    train_ratio = args.get('train_ratio', 0.99)
    name_filter = args.get('name_filter', lambda n: True)

    

    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7     # 7层解码器，中间层输出有7个，最后计算损失只取了[0]层
        nheads = 8         
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'vq': args['use_vq'],
                         'vq_class': args['vq_class'],
                         'vq_dim': args['vq_dim'],
                         'action_dim': 16,
                         'no_encoder': args['no_encoder'],
                         }
    
    elif policy_class == 'Diffusion':

        policy_config = {'lr': args['lr'],
                         'camera_names': camera_names,
                         'action_dim': 16,
                         'observation_horizon': 1,
                         'action_horizon': 8,
                         'prediction_horizon': args['chunk_size'],
                         'num_queries': args['chunk_size'],
                         'num_inference_timesteps': 10,
                         'ema_power': 0.75,
                         'vq': False,
                         }
    
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'eval_every': eval_every,
        'validate_every': validate_every,
        'save_every': save_every,
        
        'ckpt_dir': ckpt_dir,
        'ckpt_name': ckpt_name,
        'stats_name': stats_name,

        'resume_ckpt_path': resume_ckpt_path,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'pretrain_path': args['pretrain_path'],
        'use_base_robot': args['use_base_robot']
    }

   

   
   
    # save dataset stats 统计值
 
    success, _ = eval_bc(config)


def eval_bc(config):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    stats_name = config['stats_name']
    ckpt_name = config['ckpt_name']

    state_dim = config['state_dim']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    temporal_agg = config['temporal_agg']
    use_base_robot = config['use_base_robot']
    
    # 1 load policy and stats
    ## 1.1 create model
    policy = make_policy(policy_class, policy_config)
    
    ## 1.2 load model
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    loading_status = policy.deserialize(torch.load(ckpt_path))
    
    print(loading_status)
    
    policy.cuda()
    policy.eval()

    print(f'Loaded: {ckpt_path}')
    
    ## 1.3 load stats
    stats_path = os.path.join(ckpt_dir, stats_name)
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    # 2 preprocess预处理 
    ## 2.1 qpos归一化
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    
    ## 2.2 acticns归一化
    if policy_class == 'Diffusion':
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
    else:
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    query_frequency = policy_config['num_queries']
    
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']
    
    ### evaluation loop 
    with torch.inference_mode():
        # 一直循环推理
        while True:
            # actions初始值 [500, 532, 16]
            if temporal_agg:
                all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, 16]).cuda()
            # qpos[500, 14]
            # qpos_history_raw = np.zeros((max_timesteps, state_dim))
            
            # 每次推理都会预测max_timesteps个结果,然后组合
            for t in range(max_timesteps):

                # 1. 获取一个观察量
                obs = collections.OrderedDict()
                
                ## 1.1 图像
                image_dict = dict()
                random_image = np.random.randint(0, 256, (640, 480, 3), dtype=np.uint8)
                image_dict[config['camera_names'][0]] = random_image
                image_dict[config['camera_names'][1]] = random_image
                image_dict[config['camera_names'][2]] = random_image
                obs['images'] = image_dict

                random_matrix = np.random.rand(1, 14).reshape(-1)
                # print("random_matrix: ", random_matrix.shape)

                obs['qpos'] = random_matrix
                obs['qvel'] = random_matrix
                obs['effort'] = random_matrix
                
                if use_base_robot:
                    random_matrix = np.random.rand(1, 2).reshape(-1)
                    obs['qpos'] = np.concatenate((obs['qpos'], obs['base_vel']), axis=0)
                else:
                    obs['base_vel'] = [0.0, 0.0]

                qpos_numpy = np.array(obs['qpos'])
                # qpos_history_raw[t] = qpos_numpy
                # 预处理
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                
                if t % query_frequency == 0:
                    curr_image = get_image(obs, camera_names)
                
                # 模型推理
                if t % query_frequency == 0:
                    all_actions = policy(qpos, curr_image)   # [1, 32, 16]
                    print("all_actions: ", all_actions.shape)  

                # all_time_actions初始值全为0,形状为[500, 532, 16]
                if temporal_agg:
                    all_time_actions[[t], t:t+num_queries] = all_actions
                    print("all_time_actions: ", all_time_actions.shape)
                    actions_for_curr_step = all_time_actions[:, t]  # [500, 16]
                    print("actions_for_curr_step: ", actions_for_curr_step.shape)
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, t % query_frequency]

                
                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                
                action = post_process(raw_action)
                target_qpos = action[:-2]
                base_action = action[-2:]
                print("target_qpos: ", target_qpos)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--ckpt_name', action='store', type=str, default="policy_best.ckpt", help='--ckpt_name', required=False)
    parser.add_argument('--stats_name', action='store', type=str, default="dataset_stats.pkl", help='--stats_name', required=False)
    
    parser.add_argument('--episode_len', action='store', type=int, default=500, help='episode_len', required=False)
    parser.add_argument('--task_name', action='store', type=str, default="aloha_mobile_dummy", help='task_name', required=False)
    parser.add_argument('--camera_names', action='store', type=list,
                        default=['cam_high', 'cam_left_wrist', 'cam_right_wrist'], help='camera_names', required=False)

    parser.add_argument('--eval', action='store_true')
   
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # 预训练权重
    parser.add_argument('--pretrain_path', action='store', type=str, help="~/ckpt_dir01/policy_last.ckpt", required=False)
    parser.add_argument('--resume_ckpt_path', action='store', type=str, help='resume_ckpt_path', required=False)

    parser.add_argument('--eval_every', action='store', type=int, default=1, help='eval_every', required=False)
    parser.add_argument('--validate_every', action='store', type=int, default=1, help='validate_every', required=False)
    parser.add_argument('--save_every', action='store', type=int, default=100, help='save_every', required=False)

    parser.add_argument('--skip_mirrored_data', action='store_true')
    parser.add_argument('--actuator_network_dir', action='store', type=str, help='actuator_network_dir', required=False)
    parser.add_argument('--history_len', action='store', type=int)
    parser.add_argument('--future_len', action='store', type=int)
    parser.add_argument('--prediction_len', action='store', type=int)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    
    parser.add_argument('--use_vq', action='store_true') # 是否用vq
    parser.add_argument('--vq_class', action='store', type=int, help='vq_class')
    parser.add_argument('--vq_dim', action='store', type=int, help='vq_dim')
    parser.add_argument('--no_encoder', action='store_true')
    
    parser.add_argument('--use_base_robot', action='store_true') # 是否用vq

    main(vars(parser.parse_args()))

    # python3 imitate_episodes.py --task_name aloha_mobile_dummy --ckpt_dir ~/ckpt_dir --policy_class Diffusion --kl_weight 10 --chunk_size 32 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 --num_epochs 800  --lr 1e-5 --seed 0
    # python3 imitate_episodes.py --task_name aloha_mobile_dummy --ckpt_dir ~/ckpt_dir --policy_class ACT --kl_weight 10 --chunk_size 32 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 --num_epochs 800  --lr 1e-5 --seed 0