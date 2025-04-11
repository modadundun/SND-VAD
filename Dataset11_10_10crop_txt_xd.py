import numpy as np
from torch.utils.data import Dataset, DataLoader
import utils
import options
import os
import pickle
import random
import torch


class dataset(Dataset):
    def __init__(self, args, train=True, trainlist=None, testlist=None):
        self.args = args  # 参数
        self.dataset_path = args.dataset_path  # 数据集路径
        self.dataset_name = args.dataset_name  # 数据集名称
        self.feature_modal = args.feature_modal  # 特征模态:rgb
        self.feature_pretrain_model = args.feature_pretrain_model  # 特征预训练模型:i3d

        # self.feature_path = r"D:\anomaly\PEL4VAD-master\ucf-i3d\all"
        # print(f"Feature path: {self.feature_path}")

        # self.videoname = os.listdir(self.feature_path)
        # print(f"Video names: {self.videoname}")

        # self.data_dict = self.data_dict_creater()
        # # 更新特征路径，指向存放重命名后的.npy文件的文件夹

        # self.feature_path = r"D:\anomaly\PEL4VAD-master\ucf-i3d\all"
        # self.feature_path = r"E:\ucfclip\UCFClipFeatures"
        self.feature_path = r"E:\XD\XDClipFeatures"
        # self.feature_path = r"E:\SH_clip_video_features"
        # self.feature_path = r"D:\UCFClipFeatures_new"???

        # self.feature_path = os.path.join(self.dataset_path, self.dataset_name, 'i3d_10crop')  # 重命名后的特征路径
        self.trainlist = self.txt2list(
            txtpath=os.path.join(self.dataset_path, self.dataset_name, 'train_split_10crop.txt'))  # 训练列表
        # print(len(self.trainlist))
        # self.testlist = self.txt2list(txtpath=os.path.join(self.dataset_path, self.dataset_name, 'test_split_10crop2_filtered.txt'))  # 测试列表
        self.testlist = self.txt2list(txtpath=os.path.join(self.dataset_path, self.dataset_name, 'test_split_10crop.txt'))  # 测试列表

        self.video_label_dict = self.pickle_reader(
            file=os.path.join(self.dataset_path, self.dataset_name, 'GT', 'video_label_10crop.pickle'))  # 视频标签字典



        self.normal_video_train, self.anomaly_video_train = self.p_n_split_dataset(self.video_label_dict,
                                                                                   self.trainlist)  # 正常视频和异常视频列表

        self.train = train  # 是否为训练数据
        self.t_max = args.max_seqlen  # 最大序列长度
    def txt2list(self, txtpath=''):
        with open(file=txtpath, mode='r') as f:
            filelist = f.readlines()  # 读取文本文件的所有行
        return filelist  # 返回文本文件的列表形式

    def pickle_reader(self, file=''):
        with open(file=file, mode='rb') as f:
            video_label_dict = pickle.load(f)  # 使用pickle加载字典数据
        return video_label_dict  # 返回读取的字典数据

    def p_n_split_dataset(self, video_label_dict, trainlist):
        normal_video_train = []
        anomaly_video_train = []
        for t in trainlist:
            video_name = t.replace('\n', '').replace('Ped', 'ped')  # 处理视频名称
            if video_label_dict[video_name] == '[1.0]': #  10_crop需要
                anomaly_video_train.append(video_name)
            else:
                normal_video_train.append(video_name)

        return normal_video_train, anomaly_video_train

    def __getitem__(self, index):
        if self.train:
            normaly_indexs = random.sample(self.normal_video_train, self.args.sample_size)  # 从正常视频列表中随机采样
            anomaly_indexs = random.sample(self.anomaly_video_train, self.args.sample_size)  # 从异常视频列表中随机采样

            anomaly_features = torch.zeros(0)  # 初始化异常特征张量
            normaly_features = torch.zeros(0)  # 初始化正常特征张量
            anomaly_features_txts = torch.zeros(0)
            normal_features_txts = torch.zeros(0)
            for a_i, n_i in zip(anomaly_indexs, normaly_indexs):
                # 处理异常视频名称
                anomaly_data_video_name = a_i.replace('\n', '')
                # print(anomaly_data_video_name)
                normaly_data_video_name = n_i.replace('\n', '')  # 处理正常视频名称

                # 定义类别列表
                categories = ["_G", "G-", "B1", "B2", "B4", "B5", "B6"]

                # 存储所有匹配类别的特征
                category_features = []

                # 遍历所有类别，找到与当前异常视频名称匹配的类别
                for category in categories:
                    if category in anomaly_data_video_name:
                        npy_file_path = f'../clip_txt/txt_xd_features/{category}.npy'
                        category_features.append(np.load(npy_file_path))

                if category_features:
                    # 将所有类别特征转为PyTorch张量，并计算平均
                    anomaly_features_txt = torch.mean(torch.stack([torch.tensor(f) for f in category_features]), dim=0)

                # 处理正常视频的特征
                normal_txt_path = f'../clip_txt/txt_xd_features/label_A.npy'
                normal_features_txt = np.load(normal_txt_path)
                normal_features_txt = torch.tensor(normal_features_txt)

                # 添加一个额外的维度，使其适应批处理
                anomaly_features_txt = anomaly_features_txt.unsqueeze(0)
                normal_features_txt = normal_features_txt.unsqueeze(0)

                # 将异常视频和正常视频的特征拼接到各自的张量中
                anomaly_features_txts = torch.cat((anomaly_features_txts, anomaly_features_txt), 0)
                normal_features_txts = torch.cat((normal_features_txts, normal_features_txt), 0)
            #
            # for a_i, n_i in zip(anomaly_indexs, normaly_indexs):
            #     # 修改：使用新的文件名加载特征
            #     anomaly_data_video_name = a_i.replace('\n', '')  # 处理异常视频名称
            #     print(anomaly_data_video_name)
            #     normaly_data_video_name = n_i.replace('\n', '')  # 处理正常视频名称
            #
            #     categories = ["_G", "G-", "B1", "B2", "B4", "B5", "B6"]
            #     matched_category = None
            #     for category in categories:
            #         if category in anomaly_data_video_name:
            #             matched_category = category
            #             break
            #     if matched_category:
            #         npy_file_path = f'../clip_txt/txt_xd_features/{matched_category}.npy'
            #
            #         anomaly_features_txt = np.load(npy_file_path)
            #         anomaly_features_txt = torch.tensor(anomaly_features_txt)
            #
            #     normal_txt_path = f'../clip_txt/txt_xd_features/Normal.npy'
            #
            #     normal_features_txt = np.load(normal_txt_path)
            #     normal_features_txt = torch.tensor(normal_features_txt)
            #
            #     anomaly_features_txt = anomaly_features_txt.unsqueeze(0)
            #     normal_features_txt = normal_features_txt.unsqueeze(0)

                # anomaly_features_txts = torch.cat((anomaly_features_txts, anomaly_features_txt), 0)
                # normal_features_txts = torch.cat((normal_features_txts, normal_features_txt), 0)
                # print(normal_features_txts)

                anomaly_feature = np.load(file=os.path.join(self.feature_path, anomaly_data_video_name + '.npy'))  # 获取异常视频特征
                anomaly_feature, r = utils.process_feat_sample(anomaly_feature, self.t_max)  # 处理异常视频特征
                anomaly_feature = torch.from_numpy(anomaly_feature).unsqueeze(0)  # 转换为张量并添加维度

                normaly_feature = np.load(file=os.path.join(self.feature_path, normaly_data_video_name + '.npy'))  # 获取正常视频特征
                normaly_feature, r = utils.process_feat(normaly_feature, self.t_max, self.args.sample_step)  # 处理正常视频特征
                normaly_feature = torch.from_numpy(normaly_feature).unsqueeze(0)  # 转换为张量并添加维度

                anomaly_features = torch.cat((anomaly_features, anomaly_feature), dim=0)  # 合并异常视频特征
                normaly_features = torch.cat((normaly_features, normaly_feature), dim=0)  # 合并正常视频特征

            normaly_label = torch.zeros((self.args.sample_size, 1))  # 构造正常标签张量
            anomaly_label = torch.ones((self.args.sample_size, 1))  # 构造异常标签张量

            # print("anomaly_features:", anomaly_features.dtype)
            anomaly_features = anomaly_features.float()
            normaly_features = normaly_features.float()

            return [anomaly_features, normaly_features], [anomaly_label, normaly_label], [anomaly_features_txts, normal_features_txts]
        else:
            data_video_name = self.testlist[index].replace('\n', '')  # 获取测试视频名称
            self.feature = np.load(file=os.path.join(self.feature_path, data_video_name + '.npy'))  # 获取测试视频特征
            # print(self.feature.dtype)
            self.feature = self.feature.astype(np.float32)
            # print("self.feature_flo:",self.feature.dtype)

            return self.feature, data_video_name  # 返回特征和视频名称

    def __len__(self):
        if self.train:
            return len(self.trainlist)  # 如果是训练模式，返回训练集长度
        else:
            return len(self.testlist)  # 如果是测试模式，返回测试集长度

if __name__ == "__main__":
    args = options.parser.parse_args()  # 解析命令行参数
    train_dataset = dataset(args=args, train=True)  # 创建训练数据集对象

    train_loader = DataLoader(dataset=train_dataset, batch_size=1, pin_memory=True,
                              num_workers=0, shuffle=True)  # 创建训练数据加载器
    test_dataset = dataset(args=args, train=False)  # 创建测试数据集对象
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, pin_memory=True,
                             num_workers=0, shuffle=False)  # 创建测试数据加载器