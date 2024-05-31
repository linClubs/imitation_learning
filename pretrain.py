import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        pass
    def __forward(self):
        pass

model = MyModel()


# loading_status = model.deserialize(torch.load(config['pretrain_path']))
# self.load_state_dict(model_dict)


pretrain_path = "/home/lin/ckpt_dir/policy_last.ckpt"
model = torch.load(pretrain_path)

for k,v in model.items():
    print(k)