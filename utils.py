import numpy as np  # 导入NumPy库
import os  # 导入os模块
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块
plt.switch_backend('agg')  # 切换matplotlib绘图后端为'agg'
import zipfile  # 导入zipfile模块
import io  # 导入io模块
import torch  # 导入PyTorch库


def random_extract(feat, t_max):
    """
    随机提取特征序列中的子序列
    """
    r = np.random.randint(len(feat)-t_max)  # 随机生成子序列起始索引
    return feat[r:r+t_max], r  # 返回子序列及其起始索引


def random_extract_step(feat, t_max, step):
    """
    使用步长随机提取特征序列中的子序列
    """
    if len(feat) - step * t_max > 0:
        r = np.random.randint(len(feat) - step * t_max)  # 随机生成子序列起始索引
    else:
        r = np.random.randint(step)
    return feat[r:r+t_max:step], r  # 返回子序列及其起始索引


def random_perturb(feat, length):
    """
    随机扰动特征序列
    """
    samples = np.arange(length) * len(feat) / length  # 均匀采样索引
    for i in range(length):
        if i < length - 1:
            if int(samples[i]) != int(samples[i + 1]):
                samples[i] = np.random.choice(range(int(samples[i]), int(samples[i + 1]) + 1))
            else:
                samples[i] = int(samples[i])
        else:
            if int(samples[i]) < length - 1:
                samples[i] = np.random.choice(range(int(samples[i]), length))
            else:
                samples[i] = int(samples[i])
    return feat[samples.astype('int')], samples.astype('int')  # 返回扰动后的特征序列及采样索引


def pad(feat, min_len):
    """
    对特征序列进行填充，使其达到指定长度
    """
    if np.shape(feat)[0] <= min_len:
       return np.pad(feat, ((0, min_len-np.shape(feat)[0]), (0, 0)), mode='constant', constant_values=0)
    else:
       return feat  # 返回填充后的特征序列


def process_feat(feat, length, step):
    """
    处理特征序列，使其达到指定长度和步长
    """
    if len(feat) > length:
        if step and step > 1:
            features, r = random_extract_step(feat, length, step)  # 随机提取子序列
            return pad(features, length), r  # 返回填充后的特征序列及其起始索引
        else:
            features, r = random_extract(feat, length)  # 随机提取子序列
            return features, r  # 返回子序列及其起始索引
    else:
        return pad(feat, length), 0  # 返回填充后的特征序列及起始索引为0


def process_feat_sample(feat, length):
    """
    处理特征序列，使其达到指定长度，并进行随机扰动
    """
    if len(feat) > length:
            features, samples = random_perturb(feat, length)  # 随机扰动特征序列
            return features, samples  # 返回扰动后的特征序列及采样索引
    else:
        return pad(feat, length), 0  # 返回填充后的特征序列及起始索引为0


def scorebinary(scores=None, threshold=0.5):
    """
    将得分转换为二值化的结果
    """
    scores_threshold = scores.copy()  # 复制得分数据
    scores_threshold[scores_threshold < threshold] = 0  # 将低于阈值的得分设为0
    scores_threshold[scores_threshold >= threshold] = 1  # 将高于等于阈值的得分设为1
    return scores_threshold  # 返回二值化结果


def fill_context_mask(mask, sizes, v_mask, v_unmask):
    """
    填充变长上下文的注意力掩码
    """
    mask.fill_(v_unmask)  # 将掩码填充为未掩盖的值
    n_context = mask.size(2)  # 上下文的长度
    for i, size in enumerate(sizes):  # 对于批次中的每个上下文大小
        if size < n_context:  # 如果上下文大小小于掩码长度
            mask[i, :, size:] = v_mask  # 将超出上下文范围的位置设为掩码值
    return mask  # 返回填充后的掩码


def median(attention_logits, args):
    """
    使用中位数对注意力权重进行归一化
    """
    attention_medians = torch.zeros(0).to(args.device)  # 初始化注意力中位数
    batch_size = attention_logits.shape[0]  # 批次大小
    for i in range(batch_size):  # 对于批次中的每个样本
        attention_logit = attention_logits[i][attention_logits[i] > 0].unsqueeze(0)  # 获取有效的注意力权重
        attention_medians = torch.cat((attention_medians, attention_logit.median(1, keepdims=True)[0]), dim=0)  # 计算中位数并拼接到中位数列表中
    attention_medians = attention_medians.unsqueeze(1)  # 添加一个维度
    attention_logits_mask = attention_logits.clone()  # 克隆注意力权重
    attention_logits_mask[attention_logits <= attention_medians] = 0  # 将小于等于中位数的权重置为0
    attention_logits_mask[attention_logits > attention_medians] = 1  # 将大于中位数的权重置为1
    attention_logits = attention_logits * attention_logits_mask  # 对注意力权重进行掩码
    attention_logits_sum = attention_logits.sum(dim=2, keepdim=True)  # 计算权重的总和
    attention_logits = attention_logits / attention_logits_sum  # 归一化注意力权重
    return attention_logits  # 返回归一化后的注意力权重


# def anomap(predict_dict, label_dict, save_path, itr, save_root, zip=False, width=10, height=5):
#     """
#     生成异常检测结果的可视化图像
#     """
#     if os.path.exists(os.path.join(save_root, save_path, 'plot')) == 0:  # 如果保存路径不存在
#         os.makedirs(os.path.join(save_root, save_path, 'plot'))  # 创建保存路径
#     if zip:  # 如果需要保存为zip文件
#         zip_file_name = os.path.join(save_root, save_path, 'plot', 'itr_{}.zip'.format(itr))  # 设置zip文件名
#         with zipfile.ZipFile(zip_file_name, mode="w") as zf:  # 创建zip文件
#             for k, v in predict_dict.items():  # 遍历预测结果字典
#                 img_name = k + '.svg'  # 设置图像文件名
#                 predict_np = v.repeat(16)  # 重复预测结果以适应标签长度
#                 label_np = label_dict[k][:len(v.repeat(16))]  # 截取标签长度
#                 x = np.arange(len(predict_np))  # 生成x轴
#                 plt.plot(x, predict_np, label='Anomaly scores', color='b', linewidth=1)  # 绘制预测结果曲线
#                 plt.fill_between(x, label_np, where=label_np > 0, facecolor="red", alpha=0.3)  # 填充异常区域
#                 plt.yticks(np.arange(0, 1.1, step=0.1))  # 设置y轴刻度
#                 plt.xlabel('Frames')  # 设置x轴标签
#                 plt.grid(True, linestyle='-.')  # 添加网格线
#                 plt.legend()  # 添加图例
#                 buf = io.BytesIO()  # 创建字节流缓冲区
    #             plt.savefig(buf)  # 将图像保存到缓冲区
    #             plt.close()  # 关闭图像
    #             zf.writestr(img_name, buf.getvalue())  # 将图像写入zip文件
    # else:  # 如果不保存为zip文件
    #     for k, v in predict_dict.items():  # 遍历预测结果字典
    #         predict_np = v.repeat(16)  # 重复预测结果以适应标签长度
    #         label_np = label_dict[k][:len(v.repeat(16))]  # 截取标签长度
    #         x = np.arange(len(predict_np))  # 生成x轴
    #         plt.plot(x, predict_np, color='b', label='predicted scores', linewidth=1)  # 绘制预测结果曲线
    #         label_np = np.array(label_np)  # 确保 label_np 是 NumPy 数组
    #         plt.fill_between(x, label_np, where=label_np > 0, facecolor="r", alpha=0.3)  # 填充异常区域
    #         plt.yticks(np.arange(0, 1.1, step=0.1))  # 设置y轴刻度
    #         plt.xlabel('Frames')  # 设置x轴标签
    #         plt.ylabel('Anomaly scores')  # 设置y轴标签
    #         plt.grid(True, linestyle='-.')  # 添加网格线
    #         plt.legend()  # 添加图例
    #         if os.path.exists(os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr))) == 0:  # 如果保存路径不存在
    #             os.makedirs(os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr)))  # 创建保存路径
    #             plt.savefig(os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr), k))  # 保存图像
    #         else:
    #             plt.savefig(os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr), k))  # 保存图像
    #         plt.close()  # 关闭图像

def anomap(predict_dict, label_dict, save_path, itr, save_root, zip=False, width=15, height=5):
    """
    生成异常检测结果的可视化图像
    """
    if not os.path.exists(os.path.join(save_root, save_path, 'plot')):  # 如果保存路径不存在
        os.makedirs(os.path.join(save_root, save_path, 'plot'))  # 创建保存路径
    for k, v in predict_dict.items():  # 遍历预测结果字典
        predict_np = v.repeat(16)  # 重复预测结果以适应标签长度
        k = k[:-2]
        label_np = label_dict[k][:len(predict_np)]  # 截取标签长度
        x = np.arange(len(predict_np))  # 生成x轴
        plt.figure(figsize=(width, height))  # 设置图像大小
        plt.plot(x, predict_np, color='b', label='predicted scores', linewidth=1)  # 绘制预测结果曲线
        label_np = np.array(label_np)  # 确保 label_np 是 NumPy 数组
        # plt.fill_between(x, label_np, where=label_np > 0, facecolor="r", alpha=0.3)  # 填充异常区域
        plt.fill_between(x, label_np, where=label_np > '0', facecolor="r", alpha=0.3)  # 填充异常区域

        plt.yticks(np.arange(0, 1.1, step=0.1))  # 设置y轴刻度
        plt.xlabel('Frames')  # 设置x轴标签
        plt.ylabel('Anomaly scores')  # 设置y轴标签
        plt.grid(True, linestyle='-.')  # 添加网格线
        plt.legend()  # 添加图例
        output_dir = os.path.join(save_root, save_path, 'plot', 'itr_{}'.format(itr))
        if not os.path.exists(output_dir):  # 如果保存路径不存在
            os.makedirs(output_dir)  # 创建保存路径
        plt.savefig(os.path.join(output_dir, k + '.svg'), format='svg')  # 保存图像为SVG格式
        plt.close()  # 关闭图像


