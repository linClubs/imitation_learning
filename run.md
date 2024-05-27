
# 1 
~~~python
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
~~~





# 2 问题
~~~python
# 1 ModuleNotFoundError: No module named 'robomimic'
# 修改：
import sys
sys.path.append("./")

# 问题2
/home/agilex/cobot_magic/aloha-devel/act/detr/models/position_encoding.py:46: UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch. It currently rounds toward 0 (like the 'trunc' function NOT 'floor'). This results in incorrect rounding for negative values. To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, use torch.div(a, b, rounding_mode='floor').

  dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
# 修改：将上面这句改成下面即可
dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)
~~~