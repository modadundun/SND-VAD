import argparse
parser = argparse.ArgumentParser(description='SimiAtteen_Net')  # 创建参数解析器对象
parser.add_argument('--device', type=int, default=0, help='GPU ID')  # 添加设备参数，指定GPU ID
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate (default: 0.0001)')  # 添加学习率参数


parser.add_argument('--model_name', default='snd_vad', help=' ')  # 添加模型名称参数




parser.add_argument('--loss_type', default='MIL', type=str, help='the type of n_pair loss, max_min_2, max_min, attention, attention_median, attention_H_L or max')  # 添加损失类型参数
parser.add_argument('--pretrain', type=int, default=0)  # 添加预训练参数，指示是否进行预训练
parser.add_argument('--pretrained_ckpt', default=None, help='ckpt for pretrained model')  #
# parser.add_argument('--pretrained_ckpt', default=r"D:\anomaly\Anomaly-ucf\ckpt\modeldalstm\20240804054200\iter_3200.pkl", help='ckpt for pretrained model')  # feature_size=2048，学习率0.0001


parser.add_argument('--Lambda', type=str, default='1_5', help='')

parser.add_argument('--testing_path', type=str, default=None, help='time file for test model')  # 添加测试路径参数
parser.add_argument('--testing_model', type=str, default=None, help='iteration name for testing model')  # 添加测试模型名称参数
# parser.add_argument('--feature_size', type=int, default=2048, help='size of feature (default: 2048)')  # 添加特征大小参数
parser.add_argument('--feature_size', type=int, default=512, help='size of feature (default: 2048)')  # 添加特征大小参数

parser.add_argument('--batch_size', type=int, default=1, help='number of samples in one iteration')  # 添加批量大小参数
parser.add_argument('--sample_size', type=int, default=30, help='number of samples in one iteration')  # 添加样本大小参数
# parser.add_argument('--sample_size', type=int, default=32, help='number of samples in one iteration')  # 添加样本大小参数

parser.add_argument('--sample_step', type=int, default=1, help='')  # 添加样本步长参数

# parser.add_argument('--dataset_name', type=str, default='shanghaitech', help='')  # 添加数据集名称参数
parser.add_argument('--dataset_name', type=str, default='ucf-crime', help='')  # 添加数据集名称参数
# parser.add_argument('--dataset_name', type=str, default='xd', help='')  # 添加数据集名称参数

parser.add_argument('--dataset_path', type=str, default='../dataset', help='path to dir contains anomaly datasets')  # 添加数据集路径参数

parser.add_argument('--sh_gt', type=str, default=r'D:\anomaly\PEL4VAD-master\list\sh\sh-gt.npy', help='path to dir contains anomaly datasets')  # 添加数据集路径参数
parser.add_argument('--ucf_gt', type=str, default=r'D:\anomaly\PEL4VAD-master\list\ucf\ucf-gt.npy', help='path to dir contains anomaly datasets')  # 添加数据集路径参数


parser.add_argument('--feature_modal', type=str, default='rgb', help='features from different input, options contain rgb, flow , combine')  # 添加特征模态参数



parser.add_argument('--max-seqlen', type=int, default=300, help='maximum sequence length during training (default: 750)')  # 添加最大序列长度参数
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')  # 添加随机种子参数
parser.add_argument('--max_epoch', type=int, default=100, help='maximum iteration to train (default: 50000)')  # 添加最大训练轮数参数
parser.add_argument('--feature_pretrain_model', type=str, default='i3d', help='type of feature to be used I3D or C3D (default: I3D)')  # 添加特征预训练模型参数
parser.add_argument('--feature_layer', type=str, default='fc6', help='fc6 or fc7')  # 添加特征层参数
parser.add_argument('--k', type=int, default=5, help='value of k')  # 添加k值参数
parser.add_argument('--plot', type=int, default=1, help='whether plot the video anomalous map on testing')  # 添加绘图参数，指示是否在测试时绘制视频异常图

parser.add_argument('--larger_mem', type=int, default=0, help='') # 添加内存参数，指示是否使用更大的内存


# parser.add_argument('--confidence', type=float, default=0, help='anomaly sample threshold')
parser.add_argument('--snapshot', type=int, default=20, help='anomaly sample threshold')  # 添加快照参数，指示保存模型权重的间隔
# parser.add_argument('--ps', type=str, default='normal_loss_mean')
parser.add_argument('--label_type', type=str, default='unary')  # 添加标签类型参数

#