import os
import math
import torch
import argparse
from tqdm import tqdm
from torch import optim
from torchsummary import summary

from utils.tool import *
from utils.datasets import *
from utils.evaluation import CocoDetectionEvaluator

from module.loss import DetectorLoss
from module.detector import Detector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FastestDet:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--yaml', type=str, default="", help='.yaml config')
        parser.add_argument('--weight', type=str, default=None, help='pretrained weights path')

        opt = parser.parse_args()
        assert os.path.exists(opt.yaml), "请指定正确的配置文件路径"

        self.cfg = LoadYaml(opt.yaml)
        print(self.cfg)

        if opt.weight is not None:
            print("load weight from:%s" % opt.weight)
            self.model = Detector(self.cfg.category_num, True).to(device)
            self.model.load_state_dict(torch.load(opt.weight, map_location=device))
        else:
            self.model = Detector(self.cfg.category_num, False).to(device)

        summary(self.model, input_size=(3, self.cfg.input_height, self.cfg.input_width))

        print("use SGD optimizer")
        self.optimizer = optim.SGD(
            params=self.model.parameters(),
            lr=self.cfg.learn_rate,
            momentum=0.949,
            weight_decay=0.0005,
        )

        if self.cfg.scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.cfg.end_epoch, eta_min=self.cfg.learn_rate * 0.01
            )
        else:
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.cfg.milestones, gamma=0.1
            )

        # Automatic Mixed Precision scaler (no-op when AMP is disabled)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp)

        self.loss_function = DetectorLoss(device)
        self.evaluation = CocoDetectionEvaluator(self.cfg.names, device)

        val_dataset = TensorDataset(self.cfg.val_txt, self.cfg.input_width, self.cfg.input_height, False)
        train_dataset = TensorDataset(self.cfg.train_txt, self.cfg.input_width, self.cfg.input_height, True)

        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.cfg.num_workers,
            drop_last=False,
            persistent_workers=True,
        )
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.cfg.num_workers,
            drop_last=True,
            persistent_workers=True,
        )

    def train(self):
        batch_num = 0
        warmup_steps = self.cfg.warmup_epochs * len(self.train_dataloader)
        print('Starting training for %g epochs...' % self.cfg.end_epoch)

        for epoch in range(self.cfg.end_epoch + 1):
            self.model.train()
            pbar = tqdm(self.train_dataloader)
            for imgs, targets in pbar:
                imgs = imgs.to(device).float() / 255.0
                targets = targets.to(device)

                with torch.cuda.amp.autocast(enabled=self.cfg.amp):
                    preds = self.model(imgs)
                    iou, obj, cls, total = self.loss_function(preds, targets)

                self.scaler.scale(total).backward()

                # Gradient clipping prevents exploding gradients
                if self.cfg.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # Polynomial warmup for the first warmup_epochs epochs
                if batch_num <= warmup_steps:
                    scale = math.pow(batch_num / max(warmup_steps, 1), 4)
                    for g in self.optimizer.param_groups:
                        g['lr'] = self.cfg.learn_rate * scale

                lr = self.optimizer.param_groups[0]['lr']
                info = "Epoch:%d LR:%f IOU:%f Obj:%f Cls:%f Total:%f" % (
                    epoch, lr, iou, obj, cls, total)
                pbar.set_description(info)
                batch_num += 1

            if epoch % 10 == 0 and epoch > 0:
                self.model.eval()
                print("computing mAP...")
                mAP05 = self.evaluation.compute_map(self.val_dataloader, self.model)
                torch.save(self.model.state_dict(),
                           "checkpoint/weight_AP05:%.4f_%d-epoch.pth" % (mAP05, epoch))

            self.scheduler.step()


if __name__ == "__main__":
    model = FastestDet()
    model.train()
