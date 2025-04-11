from __future__ import print_function
import os
import torch


# from Dataset11_10_10crop_txt_sh import dataset  # 导入数据集
from Dataset11_10_10crop_txt_ucf import dataset  # 导入数据集
# from Dataset11_10_10crop_txt_xd import dataset  # 导入数据集


from torch.utils.data import DataLoader

from train_icic import train  # 导入训练函数   0.968   0.856

import options  # 导入选项配置
import torch.optim as optim
import datetime


from snd_vad2 import model_generater



if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    args = options.parser.parse_args()  # 解析命令行参数
    torch.manual_seed(args.seed)  # 设置随机种子
    # device = torch.device("cuda:{}".format(args.device))  # 设置设备类型
    device = torch.device("cuda")
    torch.cuda.set_device(args.device)  # 设置CUDA设备
    time = datetime.datetime.now()  # 获取当前时间

    # 构建保存路径
    # save_path = os.path.join(args.model_name, args.feature_pretrain_model, args.dataset_name, 'k_{}'.format(args.k), '_Lambda_{}'.format(args.Lambda), args.feature_modal, '{}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(time.year, time.month, time.day, time.hour,time.minute, time.second))
    save_path = os.path.join(args.model_name, '{}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(time.year, time.month, time.day, time.hour,time.minute, time.second))

    # 生成模型
    model = model_generater(model_name=args.model_name, feature_size=args.feature_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)  # 设置优化器
    if args.pretrained_ckpt is not None:  # 如果提供了预训练模型，则加载预训练模型
        model.load_state_dict(torch.load(args.pretrained_ckpt))

    # 加载训练和测试数据集
    # 创建训练数据集和数据加载器
    train_dataset = dataset(args=args, train=True)  # 创建训练数据集
    # print(train_dataset[:])
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, pin_memory=True,
                              num_workers=1, shuffle=True)  # 创建训练数据加载器


    # 创建测试数据集和数据加载器
    test_dataset = dataset(args=args, train=False)  # 创建测试数据集
    # train2test_dataset = dataset_train2test(args=args)  # 创建用于训练到测试的数据集
    test_loader = DataLoader(dataset=test_dataset, batch_size=10, pin_memory=True,
                             num_workers=1, shuffle=False)  # 创建测试数据加载器
    # train2test_loader = DataLoader(dataset=train2test_dataset, batch_size=1, pin_memory=True,
    #                            num_workers=2, shuffle=False)  # 创建训练到测试数据加载器
    all_test_loader = [test_loader]  # 组合测试数据加载器

    # 创建保存路径和日志文件夹
    if not os.path.exists('./ckpt/' + save_path):
        os.makedirs('./ckpt/' + save_path)

    logger = False


    train(epochs=args.max_epoch, train_loader=train_loader, all_test_loader=all_test_loader, args=args, model=model, optimizer=optimizer, logger=logger, device=device, save_path=save_path)





