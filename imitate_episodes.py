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


from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict, calibrate_linear_vel, postprocess_base_action # helper functions
from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy

from detr.models.latent_model import Latent_Model_Transformer

import IPython
e = IPython.embed
FPS = 50

# 1 读取配置文件, 创建dataloader
def main(args):
    set_seed(1)

    # TASK_CONFIGS配置
    task_name = args['task_name']
    dataset_dir = os.path.join(os.path.expanduser('~/data0308-2'), task_name)

    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
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
    }

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    # config_path = os.path.join(ckpt_dir, 'config.pkl')
    # expr_name = ckpt_dir.split('/')[-1]
    # with open(config_path, 'wb') as f:
    #     pickle.dump(config, f)
    
    # 创建dataloader
    # print("\n")
    # print(dataset_dir)
    # print(name_filter)
    # print(camera_names)
    # print(args['chunk_size'])
    # print(args['skip_mirrored_data'])
    # print(config['pretrain_path'])
    # print( args['chunk_size'], args['skip_mirrored_data'], config['pretrain_path'], policy_class, stats_dir, sample_weights, train_ratio)
    
    # EpisodicDataset继承torch.utils.data.Dataset: 返回image_data, qpos_data, action_data, is_pad
    # 产生dataloader和统计值
    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, name_filter, camera_names, batch_size_train, batch_size_val, args['chunk_size'], args['skip_mirrored_data'], config['pretrain_path'], policy_class, stats_dir_l=stats_dir, sample_weights=sample_weights, train_ratio=train_ratio)
    
    # save dataset stats 统计值
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    # 训练。返回了最小损失对应的周期、损失、模型信息
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_step, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ step{best_step}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        print("\033[32m\n Start create VAE model... \033[0m")
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        print("\033[32m\n Start create CNNMLP model... \033[0m")
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == 'Diffusion':
        print("\033[32m\n Start create Diffusion model... \033[0m")
        policy = DiffusionPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'Diffusion':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names, rand_crop_resize=False):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

    if rand_crop_resize:
        print('rand crop resize is used!')
        original_size = curr_image.shape[-2:]
        ratio = 0.95
        curr_image = curr_image[..., int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                     int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
        curr_image = curr_image.squeeze(0)
        resize_transform = transforms.Resize(original_size, antialias=True)
        curr_image = resize_transform(curr_image)
        curr_image = curr_image.unsqueeze(0)
    
    return curr_image

def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    eval_every = config['eval_every']
    validate_every = config['validate_every']
    save_every = config['save_every']

    # 设置随机种子
    set_seed(seed)

    # 创建模型
    policy = make_policy(policy_class, policy_config)
    
    # 加载预训练模型
    if config['pretrain_path'] is not None:
        loading_status = policy.deserialize(torch.load(config['pretrain_path']))
        print(f'Pretrain model Loaded {loading_status} from: {config["pretrain_path"]}')
    
    if config['resume_ckpt_path'] is not None:
        loading_status = policy.deserialize(torch.load(config['resume_ckpt_path']))
        print(f'Resume policy from: {config["resume_ckpt_path"]}, Status: {loading_status}')
    
    # 模型转到cuda-device上
    policy.cuda()
    # 创建优化器
    optimizer = make_optimizer(policy_class, policy)
    min_val_loss = np.inf  # 正的无穷大
    best_ckpt_info = None
    train_history = []
    validation_history = []

    # dataloader
    train_dataloader = repeater(train_dataloader)
    print("\n")
    # 遍历epoch # tqdm库：显示进度条
    for epoch in tqdm(range(num_epochs+1)):
        print(f'Epoch: {epoch}')
        
        print('\nValidating...')
        if epoch % validate_every == 0:  # 验证保存频率
            with torch.inference_mode():
                policy.eval()
                epoch_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy)
                    epoch_dicts.append(forward_dict)
                    if(batch_idx > 50):
                        break

                epoch_summary = compute_dict_mean(epoch_dicts)
                validation_history.append(epoch_summary)

                epoch_val_loss = epoch_summary['loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
            
            print(f'Val loss:   {epoch_val_loss:.5f}')
            summary_string = ''
            for k, v in epoch_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            print(summary_string)
            
        # training
        policy.train()       # 训练模式
        optimizer.zero_grad() # 梯度清0
        data = next(train_dataloader)  # 下个dataloader  # 取数据集

        # 前向推理
        forward_dict = forward_pass(data, policy)   # 里面会调用__get_items__遍历数据集
        # backward
        loss = forward_dict['loss'] # 得到损失值  
        loss.backward()             # 反向传播
        optimizer.step()            # 梯度更新

        # 按save_every频率保存模型
        if epoch % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.serialize(), ckpt_path)

        # 保存训练损失
        train_history.append(detach_dict(forward_dict))

        # # print(f'Val loss:   {epoch_val_loss:.5f}')
        # summary_string = ''
        # for k, v in loss.items():
        #     summary_string += f'Train loss  {k}: {v.item():.3f} '
        # print(summary_string)


    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.serialize(), ckpt_path)

    # 更新最小的损失保存一个best权重
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at step {best_epoch}')
    
    # 保存最后一次的模型和权重
    ckpt_path = os.path.join(ckpt_dir, f'policy_last.pth')
    torch.save(policy, ckpt_path)              # 保存整个模型和参数pth
    
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)
    return best_ckpt_info

def repeater(data_loader):  # 函数可以重复生成器的迭代结果
    epoch = 0
    for loader in repeat(data_loader): 
        for data in loader:
            yield data
        print(f'Epoch {epoch} done')
        epoch += 1


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in (train_history[0]):
        # 3个key, l1, kl, l1+kl
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        for key_val in (validation_history[0]):
            train_values = [summary[key].item() for summary in train_history]
            val_values = [summary[key_val].item() for summary in validation_history]

        plt.plot(np.linspace(0, num_epochs - 1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs - 1, len(validation_history)), val_values, label='val')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, default="~/data0308-2", help='dataset_dir', required=False)
    parser.add_argument('--episode_len', action='store', type=int, default=500, help='episode_len', required=False)
    parser.add_argument('--task_name', action='store', type=str, default="aloha_mobile_dummy", help='task_name', required=False)
    parser.add_argument('--camera_names', action='store', type=list,
                        default=['cam_high', 'cam_left_wrist', 'cam_right_wrist'], help='camera_names', required=False)

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    # 
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
    
    main(vars(parser.parse_args()))

    # python3 imitate_episodes.py --task_name aloha_mobile_dummy --ckpt_dir ~/ckpt_dir --policy_class Diffusion --kl_weight 10 --chunk_size 32 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 --num_epochs 800  --lr 1e-5 --seed 0
    # python3 imitate_episodes.py --task_name aloha_mobile_dummy --ckpt_dir ~/ckpt_dir --policy_class ACT --kl_weight 10 --chunk_size 32 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 --num_epochs 800  --lr 1e-5 --seed 0