
# 

episode: 回合, 记录了多少组数据 
timestep: 时间步, 一个回合数据的时长(步数)
observations: 观测值



# 1 数据采集部分

## 1.1 数据采集原理部分

1. 原本aloha没考虑数据软同步, 直接记录各个传感器当前全局变量的值, 他这个就可以从任意动作开始记录
2. timestep-obs是当前从臂(推理只有从臂)时刻的观察值, 包含****从臂、从臂gripper、image**
3. actions是记录的****主臂、或者底盘**，因为从臂是订阅的主臂的关机角度进行运动, len(actions) = 500
4. len(timestep-obs) = 501, 第一帧为初始状态，不是有主臂控制的, timestep-obs[0]与timestep-obs[i>0]的参数step_type是不一样的
5. timestep-obs和actions的len不相当, 相差1, 刚好可以对应上当前ts推理actions(可以多步)

~~~python
import dm_env
import collections  # 通用存储类型的库

# 1 先将臂移动到合适的零位置
opening_ceremony(master_bot_left, master_bot_right, env.puppet_bot_left, env.puppet_bot_right)

def get_reward():
    return 0

def get_observation():
    obs = collections.OrderedDict()
    obs['qpos'] = self.get_qpos()
    obs['qvel'] = self.get_qvel()
    obs['effort'] = self.get_effort()
    obs['images'] = self.get_images()
    obs['base_vel'] = self.get_base_vel()
    if get_tracer_vel:
        obs['tracer_vel'] = self.get_tracer_vel()
    return obs

env_reset = dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,   # 第一帧
            reward=get_reward(),
            discount=None,
            observation=get_observation())   # 从臂,gripper,图像，底盘的状态

env_obs = dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=obs)

# Data collection
ts = env_reset    # 第一帧状态
timesteps = [ts]
actions = []      # 动作第一帧为空
time0 = time.time()
DT = 1 / FPS

for t in tqdm(range(max_timesteps)):
    t0 = time.time() #
    # 1 记录主臂的joint_state
    action = get_action(master_bot_left, master_bot_right)  # 主臂的joint_state

    # 2 记录puppet, gripper, robot_base、images的state, 喂给主臂状态是为了实时更新从臂动作
    ts = env.step(action)     # 建立2个从臂, 并订阅主臂的消息, 返回当前帧的从臂joint_state、gripper、底盘数据、图像

    # 3 收集每一帧数据, 注意循环里面ts从index=1开始记录, 最终len(timesteps) = 501
    timesteps.append(ts)
    actions.append(action)
    time.sleep(max(0, DT - (time.time() - t0)))  # 等待时间
print(f'Avg fps: {max_timesteps / (time.time() - time0)}')
~~~

## 1.1 改进

1. 增加了传感器软同步部分,保证了传感器数据的实时性
2. 增加其他传感器扩展
3. python使用ros多线程同步会出问题

# 2 数据预处理

eg: 以50组episode数据, timestep_len =500步为例
1. 记录了timestep_obs(从臂的当前状态), actions(从臂的运动指令), 第一个(timestep_obs-501步)是没有action-500步
2. action: 记录的主臂或者底盘, 因此16维, `7+7+2 = 16`
3. obs: 记录了从臂`7+7 = 14`、图像`3*3*640*480`, 还包含了从臂的扭矩、速度，底盘的角速度和线速度

## 2.1 归一化处理
1. `joint_state`和`robot_base`计算每个维度上归一化,计算均值和方差,最大值和最小值
2. 在VAE模型使用了`[0,1]`归一化, Diffusion使用`[-1, 1]`归一化
3. 图像先进行`0-255`到`0-1`上缩放, 读入数据时还利用了

4. EpisodicDataset类
+ `EpisodicDataset`继承`torch.utils.data.Dataset`:
+ 重写`__init__(self):`, `__len__(self):`、`__getitem__(self, index):`三个函数
+ 在dataset的`__getitem__(self, index):`函数中,读取hdf5格式数据, 然后进行归一化, 返回值`image_data, qpos_data, action_data, is_pad`

5. 注意数据处理
500个timesteps, **随机取一个值start_st作为当前的timesteps**; actions(主臂和底盘数据)记录了从当前timesteps到chuck_size步的action,其他mask掉; image_data, qpos_data(只用了从臂的位置信息)只记录了当前timesteps的值


~~~python
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
image_data = image_data / 255.0
~~~


# 2 算法

## 2.1 VAE模型 

变分自编码器
~~~python
qpos: batch, qpos_dim
image: batch, num_cam, channel, height, width
env_state: None
actions: batch, seq, action_dim

# seq表示query, 4表示batch_size, 下面是输入维度
qpos.shape:  torch.Size([4, 14])
image.shape:  torch.Size([4, 3, 3, 480, 640])
actions.shape:  torch.Size([4, 32, 16])
is_pad.shape:  torch.Size([4, 32])


# hidden_dim=默认512
action_embed = nn.Linear(actions, hidden_dim)     # [4, 32, 512]
qpos_embed = nn.Linear(qpos, hidden_dim)          # (4, 512)
qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (4, 1, 512)

cls_embed = nn.Embedding(1, hidden_dim).weight    # [1, 512]
cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim) [4, 1, 512]

encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # [4, 1+1+32, 512]
encoder_input = encoder_input.permute(1, 0, 2)   # [34, 4, 512]


~~~


