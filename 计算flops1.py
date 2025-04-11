
import torch
from thop import profile
# from model import Model
from snd_vad import model_generater  # 导入模型生成器
# from l2fly4_2 import model_generater  # 导入模型生成器

import numpy as np
import options
args = options.parser.parse_args()  # 解析命令行参数
device = torch.device("cuda")

# 生成模型
model = model_generater(model_name=args.model_name, feature_size=args.feature_size).to(device)
model.eval()
# 创建一个输入张量，尺寸需要与你模型的输入匹配
input = torch.randn(10, 47, 512)
seq_len = torch.sum(torch.max(input.abs(), dim=2)[0] > 0, dim=1).numpy()
input = input[:, :np.max(seq_len), :]

input = input.float().to(device)

# 计算 FLOPs 和参数量
flops, params = profile(model, inputs=(input, seq_len, ))
#
# 手动将 FLOPs 和参数量换算为 G（Giga，即10^9）
flops_g = flops / 1e9
params_g = params / 1e9
#
# 打印 FLOPs 和参数量，以 G 为单位
print(f"FLOPs: {flops_g:.3f} G")
print(f"Params: {params_g:.3f} G")
# print(f"Params: {params:.3f} ")

# 模型预热
with torch.no_grad():
    for _ in range(5):  # 运行几次推理
        _ = model(input, seq_len, is_training=False)
