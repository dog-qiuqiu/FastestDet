import os
import torch
import argparse
from torchsummary import summary

from utils.tool import *
from utils.datasets import *
from utils.evaluation import CocoDetectionEvaluator

from module.detector import Detector

# 指定后端设备CUDA&CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # 指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default="", help='.yaml config')
    parser.add_argument('--weight', type=str, default=None, help='.weight config')

    opt = parser.parse_args()
    assert os.path.exists(opt.yaml), "请指定正确的配置文件路径"
    assert os.path.exists(opt.weight), "请指定正确的权重文件路径"

    # 解析yaml配置文件
    cfg = LoadYaml(opt.yaml)    
    print(cfg) 

    # 加载模型权重
    print("load weight from:%s"%opt.weight)
    model = Detector(cfg.category_num, True).to(device)
    model.load_state_dict(torch.load(opt.weight))
    model.eval()

    # # 打印网络各层的张量维度
    summary(model, input_size=(3, cfg.input_height, cfg.input_width))

    # 定义验证函数
    evaluation = CocoDetectionEvaluator(cfg.names, device)

    # 数据集加载
    val_dataset = TensorDataset(cfg.val_txt, cfg.input_width, cfg.input_height, False)

    #验证集
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=cfg.batch_size,
                                                 shuffle=False,
                                                 collate_fn=collate_fn,
                                                 num_workers=4,
                                                 drop_last=False,
                                                 persistent_workers=True
                                                 )

    # 模型评估
    print("computer mAP...")
    evaluation.compute_map(val_dataloader, model)