# --- 必要的库导入 ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import math # math 可能在某些 Positional Encoding 中使用，这里保留以防万一
import torch.nn.init as torch_init
import numpy as np # numpy 可能在数据处理中使用，这里保留

# --- 权重初始化函数 ---
def weights_init(m):
    """
    初始化模型权重。对卷积层和线性层使用 Xavier 均匀初始化，偏置项初始化为 0。
    """
    # 获取模块的类名
    classname = m.__class__.__name__
    # 如果是卷积层或线性层
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        # 使用 Xavier 均匀分布初始化权重
        torch_init.xavier_uniform_(m.weight)
        # 如果存在偏置项
        if m.bias is not None:
            # 将偏置项初始化为 0
            m.bias.data.fill_(0)

# --- 原型交互层 (替代原 NPCR 功能) ---
class PrototypeInteractionLayer(nn.Module):
    """
    原型交互层：
    将输入的特征与一组可学习的“正常原型”进行交互。
    使用基于注意力的机制计算“正常性上下文”，并将其与原始特征结合，
    输出经过正常性信息调整后的特征。
    """
    # 初始化函数
    def __init__(self, feature_dim=512, num_prototypes=5, dropout_rate=0.1):
        """
        参数:
            feature_dim (int): 输入和输出特征的维度。
            num_prototypes (int): 可学习的正常原型的数量。
            dropout_rate (float): Dropout 比率。
        """
        # 调用父类初始化
        super().__init__()
        # 定义可学习的正常原型 (作为 Attention 的 'Key')
        self.normal_prototypes_keys = nn.Parameter(torch.randn(num_prototypes, feature_dim))
        # 定义与原型相关联的值 (作为 Attention 的 'Value')
        self.normal_prototypes_values = nn.Parameter(torch.randn(num_prototypes, feature_dim))
        # 定义缩放因子，用于 Attention 计算
        self.scale = feature_dim ** -0.5
        # 定义 Layer Normalization，用于稳定训练
        self.norm = nn.LayerNorm(feature_dim)
        # 定义一个输出投影层（也可以是更复杂的 MLP）
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        # 定义 GELU 激活函数
        self.activation = nn.GELU()
        # 定义 Dropout 层
        self.dropout = nn.Dropout(dropout_rate)

        # 初始化原型参数 (可选，但通常推荐)
        torch_init.normal_(self.normal_prototypes_keys, std=0.02)
        torch_init.normal_(self.normal_prototypes_values, std=0.02)

    # 定义前向传播
    def forward(self, features):
        """
        参数:
            features (torch.Tensor): 输入的实例特征张量，形状 [B, T, D]
        返回:
            torch.Tensor: 经过原型交互调整后的特征，形状 [B, T, D]
        """
        # features: 输入特征 [B, T, D]
        B, T, D = features.shape
        # N: 原型数量
        N = self.normal_prototypes_keys.shape[0]
        # print(self.normal_prototypes_keys.shape)  # torch.Size([5, 512])

        # 1. 计算输入特征 (Query) 与原型 (Key) 的相似度 (缩放点积)
        attn_logits = torch.matmul(features, self.normal_prototypes_keys.t()) * self.scale # [B, T, N]
        # print("attn_logits:", attn_logits.shape)  # ([60, 257, 5])
        # 2. 计算注意力权重 (Softmax over prototypes)
        attn_weights = F.softmax(attn_logits, dim=-1) # [B, T, N]
        # print("attn_weights:", attn_weights.shape)  # ([60, 257, 5])
        # 3. 计算“正常性上下文” (加权求和原型值)
        normality_context = torch.matmul(attn_weights, self.normal_prototypes_values)  # [B, T, D]
        # print(520)
        # print(self.normal_prototypes_values.shape)  # torch.Size([5, 512])
        # print("normality_context:",normality_context.shape)  # ([60, 257, 512])
        # 4. 结合原始特征与正常性上下文 (使用残差连接)
        combined_features = features + normality_context  # [B, T, D]
        # print("combined_features:", combined_features.shape)  # ([60, 257, 512])
        # 5. 应用 Layer Normalization
        normed_features = self.norm(combined_features)
        # print("normed_features:", normed_features.shape)  # ([60, 257, 512])
        # 6. 应用输出投影和激活函数
        projected_features = self.output_proj(normed_features) # [B, T, D]
        # print("projected_features:", projected_features.shape)  # ([60, 257, 512])
        activated_features = self.activation(projected_features) # [B, T, D]
        # print("activated_features:", activated_features.shape)  # ([60, 257, 512])

        # 7. 应用 Dropout
        output_features = self.dropout(activated_features) # [B, T, D]
        # print("output_features:", output_features.shape)  # ([60, 257, 512])

        # 返回最终调整后的特征
        return output_features

# --- 伪实例区分性增强模块 (保留为辅助损失) ---
class PseudoInstanceDiscriminativeEnhancement(nn.Module):
    """
    伪实例区分性增强 (PIDE)：
    根据实例分数生成伪标签（异常/正常），并应用对比学习损失，
    旨在推开伪异常和伪正常实例的特征表示。
    在这个版本中，它作为一个可选的辅助损失函数使用。
    """
    # 初始化函数
    def __init__(self, temperature=0.1):
        # 调用父类初始化
        super().__init__()
        # 设置温度系数
        self.temperature = temperature

    # 定义前向传播函数
    # features: 输入的实例特征张量 (形状 [B, T, D])
    # element_logits: 实例级的分数 (经过 Sigmoid 后，范围 [0, 1]) (形状 [B, T, 1])
    # seq_len: 每个序列的实际有效长度列表或张量 (形状 [B])
    def forward(self, features, element_scores, seq_len):
        """
        计算 PIDE 辅助损失。
        注意：这里输入的 element_scores 应该是 sigmoid 后的分数。
        """
        # 获取输入的形状信息
        batch_size, max_seq_len, feature_dim = features.shape

        # 初始化伪标签张量 (0: 未标记, 1: 伪异常, -1: 伪正常)
        pseudo_labels = torch.zeros(batch_size, max_seq_len, device=features.device) # 形状 [B, T]

        # 遍历批次生成伪标签 (使用 argmax/argmin 策略)
        for i in range(batch_size):
            # 获取有效长度和对应的分数
            valid_len = seq_len[i].item() # .item() 将 tensor 转为 python int
            if valid_len > 0:
                valid_scores = element_scores[i, :valid_len, 0] # 取出有效分数 [L]
                # 找到分数最高和最低的索引
                max_idx = torch.argmax(valid_scores)
                min_idx = torch.argmin(valid_scores)
                # 标记伪标签
                pseudo_labels[i, max_idx] = 1
                pseudo_labels[i, min_idx] = -1

        # 特征归一化
        normalized_features = F.normalize(features, p=2, dim=-1) # [B, T, D]
        # print("normalized_features:", normalized_features.shape)  # ([60, 257, 512])

        # 计算相似度矩阵 (批次内两两实例)
        # (B, T, D) @ (B, D, T) -> (B, T, T)
        sim_matrix = torch.matmul(normalized_features, normalized_features.transpose(-2, -1)) / self.temperature
        # print("sim_matrix:", sim_matrix.shape)  # ([60, 257, 257])

        # 生成对比损失的掩码
        # mask > 0 表示同为伪异常或同为伪正常的配对 ((+,+), (-,-))
        mask = pseudo_labels.unsqueeze(1) * pseudo_labels.unsqueeze(2)  # [B, T, T]
        # print("mask:", mask.shape)  # ([60, 257, 257])

        # 计算 InfoNCE 形式的对比损失
        exp_sim = torch.exp(sim_matrix)
        # 对角线元素通常需要移除或特殊处理，这里简化，假设不影响主要梯度
        # log( exp(sim_ij) / sum_k(exp(sim_ik)) )
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=-1, keepdim=True) + 1e-8)  # [B, T, T]
        # print("log_prob:", log_prob.shape)  # ([60, 257, 257])

        # 只关注正样本对 ((+,+) 或 (-,-) 对)
        mask_positive_pairs = (mask > 0).float()  # [B, T, T]
        # print("mask_positive_pairs:", mask_positive_pairs.shape)  # ([60, 257, 257])

        # 计算损失：最大化正样本对的 log_prob (即最小化其负值)
        # 对所有正样本对的 log_prob 求和并取负，然后平均
        loss = - (mask_positive_pairs * log_prob).sum() / (mask_positive_pairs.sum() + 1e-8)

        # 返回计算出的 PIDE 损失值
        return loss


# --- 集成后的主模型 SND_VAD ---
class SND_VAD(nn.Module):
    """
    SND_VAD 模型：
    集成了 PrototypeInteractionLayer 来处理输入特征，
    并保留 PIDE 作为可选的辅助损失。
    """
    # 初始化函数
    def __init__(self, feature_size=512):
        # 调用父类初始化
        super().__init__()
        # 使用新的原型交互层处理输入特征
        self.prototype_layer = PrototypeInteractionLayer(feature_dim=feature_size)
        # 主要的多层感知机分类器，用于实例级评分
        self.classifier = nn.Sequential(
            # 输入维度现在仍是 feature_size，因为 prototype_layer 输出维度不变
            nn.Linear(feature_size, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1)  # 输出原始逻辑值 (raw logits)
        )
        # Sigmoid 函数，将逻辑值转换为 [0, 1] 的分数
        self.sigmoid = nn.Sigmoid()
        # Dropout 层，用于训练时正则化
        self.dropout = nn.Dropout(0.5) # Dropout 通常放在激活函数之后，或者像这里，在特征处理后应用

        # 实例化 PIDE 辅助损失模块 (NPCR 模块已移除)
        self.pide = PseudoInstanceDiscriminativeEnhancement()

        # 应用权重初始化
        self.apply(weights_init)

    # 定义前向传播函数
    # x: 输入特征 [B, T, D]
    # seq_len: 每个序列的有效长度 [B]
    # is_training: 是否处于训练模式
    def forward(self, x, seq_len=None, is_training=True):
        """
        模型的前向传播过程。
        """
        # 1. 特征通过原型交互层进行处理和调整
        processed_features = self.prototype_layer(x) # [B, T, D]
        # print("processed_features:", processed_features.shape)

        # 2. (可选) 在训练时应用 Dropout
        # 注意：PrototypeInteractionLayer 内部已有 Dropout，这里可以酌情添加或移除
        # if is_training:
        #     processed_features = self.dropout(processed_features)

        # 保存处理后的特征，可能用于 PIDE 损失计算或后续分析
        visual_features = processed_features  # [B, T, D]
        # print("visual_features:", visual_features.shape)

        # 3. 通过分类器获取实例级的原始逻辑值 (raw logits)
        raw_logits = self.classifier(visual_features) # [B, T, 1]
        # print("raw_logits:", raw_logits.shape)

        # 4. 通过 Sigmoid 函数获取最终的实例级异常分数
        element_scores = self.sigmoid(raw_logits) # [B, T, 1]
        # print("element_scores:", element_scores.shape)

        # 5. 如果是训练模式，计算 PIDE 辅助损失
        if is_training:
            # 确保 seq_len 是存在的，如果模型设计需要它
            if seq_len is None:
                 # 如果没提供 seq_len，则假设所有序列都是满长度 T
                 # 注意：这可能不适用于变长序列！实际应用中应确保提供正确的 seq_len
                 seq_len = torch.full((x.size(0),), x.size(1), dtype=torch.long, device=x.device)

            # 计算 PIDE 损失，使用处理后的特征和最终分数
            pide_loss = self.pide(visual_features, element_scores, seq_len)

            # 返回：原始逻辑值, sigmoid分数, 交互层输出的特征, PIDE损失
            return raw_logits, element_scores, visual_features, pide_loss
        else:
            # 推理模式，不计算辅助损失
            # 返回：原始逻辑值, sigmoid分数, 交互层输出的特征
            return raw_logits, element_scores, visual_features

# --- 模型生成器函数 ---
def model_generater(model_name, feature_size):
    """
    根据名称和特征大小生成模型实例。
    """
    # 如果请求的是 'snd_vad' 模型
    if model_name == 'snd_vad':
        # 返回我们新定义的 SND_VAD 模型实例
        return SND_VAD(feature_size=feature_size)
    # 否则，抛出错误
    else:
        raise ValueError(f"不支持的模型名称: {model_name}")

