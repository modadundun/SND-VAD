import pickle  # 导入pickle模块，用于数据的序列化和反序列化操作
import os  # 导入os模块，用于与操作系统进行交互
import numpy as np  # 导入numpy模块，用于科学计算
from sklearn.metrics import roc_auc_score, confusion_matrix  # 从sklearn.metrics模块导入roc_auc_score函数和confusion_matrix函数
# import sys  # 导入sys模块，用于访问与Python解释器相关的变量和函数
from utils import scorebinary, anomap  # 从自定义模块utils中导入scorebinary函数和anomap函数
from sklearn.metrics import auc, roc_curve, confusion_matrix, precision_recall_curve


def eval_p(itr, dataset, predict_dict, logger, save_path, args, plot=False, zip=False, manual=False):

    global label_dict_path  # 全局变量label_dict_path
    if manual:  # 如果manual参数为True
        save_root = './manul_test_result'  # 将save_root设置为'./manul_test_result'
    else:  # 否则
        save_root = './result'  # 将save_root设置为'./result'

    if dataset == 'shanghaitech':  # 如果dataset参数为'shanghaitech'
        label_dict_path = '{}/shanghaitech/GT'.format(args.dataset_path)  # 根据args.dataset_path拼接出label_dict_path
        with open(file=os.path.join(label_dict_path, 'frame_label.pickle'), mode='rb') as f:  # 打开文件，读取帧标签数据
            frame_label_dict = pickle.load(f)  # 使用pickle模块加载帧标签数据
        with open(file=os.path.join(label_dict_path, 'video_label_10crop.pickle'),
                  mode='rb') as f:  # 打开shanhaitech文件，读取视频标签数据
            video_label_dict = pickle.load(f)  # 使用pickle模块加载视频标签数据
        all_predict_np = np.zeros(0)  # 创建全零数组all_predict_np
        all_label_np = np.zeros(0)  # 创建全零数组all_label_np
        for k, v in predict_dict.items():  # 遍历预测字典
            base_video_name = k[:-2]
            if video_label_dict[k] == '[1.0]':  # 如果视频标签为[1.]
                frame_labels = frame_label_dict.get(base_video_name, None)
                all_predict_np = np.concatenate((all_predict_np, v.repeat(16)))  # 将预测结果扩展并连接到all_predict_np
                all_label_np = np.concatenate((all_label_np, frame_labels[:len(v.repeat(16))]))  # 将帧标签连接到all_label_np
            elif video_label_dict[k] == '[0.0]':  # 如果视频标签为[0.]
                frame_labels = frame_label_dict.get(base_video_name, None)
                all_predict_np = np.concatenate((all_predict_np, v.repeat(16)))  # 将预测结果扩展并连接到all_predict_np
                all_label_np = np.concatenate((all_label_np, frame_labels[:len(v.repeat(16))]))  # 将帧标签连接到all_label_np
        all_auc_score = roc_auc_score(y_true=all_label_np, y_score=all_predict_np)  # 计算所有视频的AUC分数
        print('Iteration: {} Area Under the Curve is {}'.format(itr, all_auc_score))  # 打印所有视频的AUC分数
        if plot:  # 如果plot参数为True
            anomap(predict_dict, frame_label_dict, save_path, itr, save_root, zip, width=15, height=5)  # 绘制异常检测地图
        if os.path.exists(os.path.join(save_root, save_path)) == 0:  # 如果目录不存在
            os.makedirs(os.path.join(save_root, save_path))  # 创建目录
        with open(file=os.path.join(save_root, save_path, 'result.txt'), mode='a+') as f:  # 打开文件，追加写入结果
            f.write('itration_{}_AUC is {}\n'.format(itr, all_auc_score))  # 写入所有视频的AUC分数




    if dataset == 'ucf-crime':  # 如果dataset参数为'shanghaitech'
        label_dict_path = '{}/ucf-crime/GT'.format(args.dataset_path)  # 根据args.dataset_path拼接出label_dict_path
        with open(file=os.path.join(label_dict_path, 'ucf_gt_upgate.pickle'), mode='rb') as f:  # 打开文件，读取帧标签数据
            frame_label_dict = pickle.load(f)  # 使用pickle模块加载帧标签数据
        with open(file=os.path.join(label_dict_path, 'video_label_10crop.pickle'),
                  mode='rb') as f:  # 打开shanhaitech文件，读取视频标签数据
            video_label_dict = pickle.load(f)  # 使用pickle模块加载视频标签数据
        all_predict_np = np.zeros(0)  # 创建全零数组all_predict_np
        all_label_np = np.zeros(0)  # 创建全零数组all_label_np
        for k, v in predict_dict.items():  # 遍历预测字典
            base_video_name = k[:-2]
            if video_label_dict[k] == '[1.0]':  # 如果视频标签为[1.]
                frame_labels = frame_label_dict.get(base_video_name, None)
                all_predict_np = np.concatenate((all_predict_np, v.repeat(16)))  # 将预测结果扩展并连接到all_predict_np
                all_label_np = np.concatenate((all_label_np, frame_labels[:len(v.repeat(16))]))  # 将帧标签连接到all_label_np
            elif video_label_dict[k] == '[0.0]':  # 如果视频标签为[0.]
                frame_labels = frame_label_dict.get(base_video_name, None)
                all_predict_np = np.concatenate((all_predict_np, v.repeat(16)))  # 将预测结果扩展并连接到all_predict_np
                all_label_np = np.concatenate((all_label_np, frame_labels[:len(v.repeat(16))]))  # 将帧标签连接到all_label_np
        all_auc_score = roc_auc_score(y_true=all_label_np, y_score=all_predict_np)  # 计算所有视频的AUC分数
        print('Iteration: {} Area Under the Curve is {}'.format(itr, all_auc_score))  # 打印所有视频的AUC分数
        if plot:  # 如果plot参数为True
            anomap(predict_dict, frame_label_dict, save_path, itr, save_root, zip, width=15, height=5)  # 绘制异常检测地图
        if os.path.exists(os.path.join(save_root, save_path)) == 0:  # 如果目录不存在
            os.makedirs(os.path.join(save_root, save_path))  # 创建目录
        with open(file=os.path.join(save_root, save_path, 'result.txt'), mode='a+') as f:  # 打开文件，追加写入结果
            f.write('itration_{}_AUC is {}\n'.format(itr, all_auc_score))  # 写入所有视频的AUC分数



    if dataset == 'xd':  # 如果dataset参数为'shanghaitech'
        label_dict_path = '{}/xd/GT'.format(args.dataset_path)  # 根据args.dataset_path拼接出label_dict_path
        all_predict_np = np.zeros(0)  # 创建全零数组all_predict_np
        gt = np.load(r'D:\anomaly\Anomaly-ucf\dataset\xd\GT\xd_gt.npy')
        gt = list(gt)
        for k, v in predict_dict.items():  # 遍历预测字典
            all_predict_np = np.concatenate((all_predict_np, v.repeat(16)))  # 将预测结果扩展并连接到all_predict_np
        all_auc_score = roc_auc_score(y_true=gt, y_score=all_predict_np)  # 计算所有视频的AUC分数
        pre, rec, _ = precision_recall_curve(list(gt), all_predict_np)
        pr_auc = auc(rec, pre)
        print(pr_auc)
        print('Iteration: {} Area Under the Curve is {}'.format(itr, all_auc_score))  # 打印所有视频的AUC分数
        if plot:  # 如果plot参数为True
            anomap(predict_dict, frame_label_dict, save_path, itr, save_root, zip, width=15, height=5)  # 绘制异常检测地图
        if os.path.exists(os.path.join(save_root, save_path)) == 0:  # 如果目录不存在
            os.makedirs(os.path.join(save_root, save_path))  # 创建目录
        with open(file=os.path.join(save_root, save_path, 'result.txt'), mode='a+') as f:  # 打开文件，追加写入结果
            f.write('itration_{}_AUC is {}\n'.format(itr, all_auc_score))  # 写入所有视频的AUC分数
