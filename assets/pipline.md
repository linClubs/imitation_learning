
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

## 1.2 改进

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


# 3 算法

## 3.1 VAE模型 

+ 通过(图像**resnet特征层**、qpos**一层线性层**)和(qpos与action只**编码器**得到潜在特征)，使用Transformer得到预测的**序列actions**, 和潜在特征的均值和方差


1. encode部分

+ 只编码action和qpos+位置编码, 生成潜在空间的特征和`mu，log(var)`
+ 最后潜在空间的特征通过线性层映射回去`latent_input[4, 512]`
+ 返回浅层空间`latent_input[4, 512]`、`mu[4, 32]`、`log(var)[4, 32]`

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

# hidden_dim=默认512  输入是action,qpos,词嵌入组合后得到encoder_input：[34, 4, 512]
action_embed = nn.Linear(actions, hidden_dim)     # [4, 32, 512]
qpos_embed = nn.Linear(qpos, hidden_dim)          # (4, 512)
qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (4, 1, 512)

cls_embed = nn.Embedding(1, hidden_dim).weight    # [1, 512]
cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim) [4, 1, 512]

encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # [4, 1+1+32, 512]
encoder_input = encoder_input.permute(1, 0, 2)   # [34, 4, 512]

# 只用了transform的编码层, 输入是encoder_input[34, 4, 512], 输出encoder_output统计所有的中间层输出
encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
# print("encoder_output: ", encoder_output.shape)

# 只取编码层输出的的第一层encoder_output[0]作为结果,维度变成[4, 512]
encoder_output = encoder_output[0] # take cls output only  # 组合时cls_embed排在0位
# 然后预测2个值，一个mu,一个log(std)
# self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2)
latent_info = self.latent_proj(encoder_output)  # [4, 64]

mu = latent_info[:, :self.latent_dim]          # 均值[4, 32]
logvar = latent_info[:, self.latent_dim:]      # 方差[4, 32]
latent_sample = reparametrize(mu, logvar)  # 重参数化 生成浅层空间特征[4, 32]
latent_input = self.latent_out_proj(latent_sample) # 线性层映射回去[4, 512]
return latent_input, probs, binaries, mu, logvar
~~~

2. 图像部分
+ n个摄像头建立了n个backbone， 这里是3个
+ 位置编码2种,一种是可学习的, 一种是三角函数
+ 通过resnet网络输入图像`[4, 3, 3, 480, 640]`,输出`[4, 3, 3, 15, 20]`和图像位置编码
~~~python
# 位置编码
def build_position_encoding(args):
    N_steps = args.hidden_dim // 2                 # 隐藏层512。N_steps=256
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")
    return position_embedding


# backbone
def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0   # True
    return_interm_layers = args.masks       # 
    # 主干网络用的ResNet, 冻结了Batch参数, 返回的最后一层数据, 
    # train_backbone最后网络中没有使用这个参数
    backbone_output = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone_output, position_embedding)
    model.num_channels = backbone_output.num_channels
    return model

# 通过name获得搭建预训练模型 resnet18
backbone = getattr(torchvision.models, name)(
        # 将第1层、第2层使用原卷积,第3层替换成扩张卷积
        replace_stride_with_dilation=[False, False, dilation],
        # 否使用预训练的权重
        # is_main_process()判断当前进程是否是主进程（在多GPU或分布式训练中），
        # 以确保只有主进程下载和加载预训练权重，其他进程不重复下载
        pretrained=is_main_process(), 
        # 将模型中的批归一化层（Batch Normalization layer）替换为冻结的批归一化层（Frozen BatchNorm）
        norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm

~~~

3. act的VAE模型的forwad函数

+ 输入self.transformer(图像`[4, 512, 15, 60]`,掩码`None`, `query`词嵌入`[32, 512]`，图像位置编码`[1, 512, 15, 60]`, 潜层特征`[34, 4, 512]`, qpos的线性输出`[4, 512]`), 经过`Transformer`层后输出(保留中间层)`[7, 4, 32, 512]`

+ 经过Transformer后, 取的多头的第一层`[4, 32, 512]`作为Tr的输出，最后通过2个线性层得到输出`[4, 32, 16]`, 填充`[4, 32, 1]`

+ 这里得到的`[4, 32, 16]`表示模型的前向输出, 表示预测了32个连续的action
+ 最终forward返回值为`预测的acticons：a_hat[4, 32, 16], is_pad_hat[4, 32, 1], 均值和方差[mu(4, 32), logvar(4, 32)], probs(None), binaries(None)`


~~~python
# forwad中图像处理函数
all_cam_features = []
all_cam_pos = []
for cam_id, cam_name in enumerate(self.camera_names):
    # 单张图的shape： image[:, cam_id].shape = [4, 3, 480, 640]
    features, pos = self.backbones[cam_id](image[:, cam_id])
    # features的维度[4, 512, 15, 20] ,640, 320, 160, 80, 40, 20 卷积了5次.索引从0开始,返回的第4次卷积结果就是20
    features = features[0] # take the last layer feature  [4, 512, 15, 20]
    pos = pos[0]                                        # [1, 512, 15, 20]
    all_cam_features.append(self.input_proj(features))  # 调整resNet的结果维度
    all_cam_pos.append(pos)

proprio_input = self.input_proj_robot_state(qpos)   #[4, 14] -> [4, 512]
    
# fold camera dimension into width dimension
# 组合图像，宽维度上(最后一个维度)
src = torch.cat(all_cam_features, axis=3)          # 组合图像    [4, 512, 15, 60]
pos = torch.cat(all_cam_pos, axis=3)               # 组合位置编码 [1, 512, 15, 60]

# 调用的Transformer类的forward函数

# def forward(self, src, mask, query_embed, pos_embed, latent_input=None, proprio_input=None, additional_pos_embed=None):
# def forward(self, 图像, 掩码None, query词嵌入, 图像的位置编码, 潜层特征(通过encode得到[34, 4, 512]), pose线性层出来后的结果[4, 512])

# 词嵌入query_embed = nn.Embedding(num_queries, hidden_dim) => self.query_embed.weight.shape=[32, 512]

# 输入self.transformer(图像[4, 512, 15, 60],掩码None, query词嵌入[32, 512]，图像位置编码[1, 512, 15, 60], 潜层特征[34, 4, 512], pose的线性输出[4, 512])
hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
# 输出hs.shape[7, 4, 32, 512], hs[0].shape[4, 32, 512]

a_hat = self.action_head(hs)  # [4, 32, 16]    # 线性层[4, 32, 512]->[4, 32, 16]
is_pad_hat = self.is_pad_head(hs) # [4, 32, 1] # 线性层[4, 32, 512]->[4, 32, 1]
~~~

4. Transformer结构
~~~python
Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        ）
~~~

# 4 VAE损失函数


+ 通过(图像**resnet特征层**)、(qpos**一层线性层**)、(qpos与action只**编码器**得到潜在特征)，三个(加上位置编码)喂给Transformer得到预测的**序列actions**, 并返回序列actions、潜在特征的均值、方差

+ 重建损失和KL散度

+ 通过模型预测的序列actions[4, 32, 16] - 原始的actions[4, 32, 16]即可
+ 这里采用L1，L2都可以

+ 训练时`policy.py`中计算损失`def __call__()`函数
~~~python
# 1 KL散度
klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
total_kld = klds.sum(1).mean(0, True)
dimension_wise_kld = klds.mean(0)
mean_kld = klds.mean(1).mean(0, True)

# 2 重建损失L1
all_l1 = F.l1_loss(actions, a_hat, reduction='none')
# 取非掩码部分
l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()

# 总损失, kl和l1量纲不一样, kl乘个系数
loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
~~~


# 5 推理过程inference process

+ `VAE`模型只喂了`image`和`qpos`给模型然后调用`model()`, `actions`给的`None`值， `forward`函数调用与训练时不同的

