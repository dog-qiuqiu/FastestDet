import math
import torch
import torch.nn as nn


class DetectorLoss(nn.Module):
    def __init__(self, device):
        super(DetectorLoss, self).__init__()
        self.device = device

    def bbox_iou(self, box1, box2, eps=1e-7):
        # Returns SIoU of box1 to box2. box1 is 4, box2 is nx4
        # https://arxiv.org/pdf/2205.12740.pdf
        box1 = box1.t()
        box2 = box2.t()

        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps
        iou = inter / union

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)

        # SIoU: Angle cost — sin(2α) = 1 - cos(2α - π/2), simplified via identity
        # sin(2·arcsin(x)) = 2x√(1-x²), but paper uses cos(2·arcsin(x) - π/2) = sin(2·arcsin(x))
        s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
        s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
        sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
        sin_alpha_1 = torch.abs(s_cw) / (sigma + eps)
        sin_alpha_2 = torch.abs(s_ch) / (sigma + eps)
        # Select the smaller angle (≤ 45°) to ensure stable convergence
        threshold = pow(2, 0.5) / 2
        sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
        # cos(2·arcsin(x) - π/2) = sin(2·arcsin(x)) = 2·sin(arcsin(x))·cos(arcsin(x)) = 2x√(1-x²)
        angle_cost = 1 - 2 * torch.pow(torch.sin(torch.arcsin(sin_alpha)), 2)

        # SIoU: Distance cost
        rho_x = (s_cw / (cw + eps)) ** 2
        rho_y = (s_ch / (ch + eps)) ** 2
        gamma = angle_cost - 2
        distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)

        # SIoU: Shape cost
        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)

        iou = iou - 0.5 * (distance_cost + shape_cost)
        return iou

    def build_target(self, preds, targets):
        N, C, H, W = preds.shape
        gt_box, gt_cls, ps_index = [], [], []
        # Each grid cell uses 4 quadrant offsets for cross-grid matching
        quadrant = torch.tensor([[0, 0], [1, 0],
                                 [0, 1], [1, 1]], device=self.device)

        if targets.shape[0] > 0:
            scale = torch.ones(6).to(self.device)
            scale[2:] = torch.tensor(preds.shape)[[3, 2, 3, 2]]
            gt = targets * scale

            gt = gt.repeat(4, 1, 1)

            quadrant = quadrant.repeat(gt.size(1), 1, 1).permute(1, 0, 2)
            gij = gt[..., 2:4].long() + quadrant
            j = torch.where(gij < H, gij, 0).min(dim=-1)[0] > 0

            gi, gj = gij[j].T
            batch_index = gt[..., 0].long()[j]
            ps_index.append((batch_index, gi, gj))

            gbox = gt[..., 2:][j]
            gt_box.append(gbox)

            gt_cls.append(gt[..., 1].long()[j])

        return gt_box, gt_cls, ps_index

    def forward(self, preds, targets):
        ft = torch.cuda.FloatTensor if preds[0].is_cuda else torch.Tensor
        cls_loss, iou_loss, obj_loss = ft([0]), ft([0]), ft([0])

        cls_criterion = nn.CrossEntropyLoss()
        obj_criterion = nn.SmoothL1Loss(reduction='none')

        gt_box, gt_cls, ps_index = self.build_target(preds, targets)

        pred = preds.permute(0, 2, 3, 1)
        pobj = pred[:, :, :, 0]
        preg = pred[:, :, :, 1:5]
        pcls = pred[:, :, :, 5:]

        N, H, W, C = pred.shape
        tobj = torch.zeros_like(pobj)
        factor = torch.ones_like(pobj) * 0.75

        if len(gt_box) > 0:
            b, gx, gy = ps_index[0]
            ptbox = torch.ones((preg[b, gy, gx].shape)).to(self.device)
            ptbox[:, 0] = preg[b, gy, gx][:, 0].tanh() + gx
            ptbox[:, 1] = preg[b, gy, gx][:, 1].tanh() + gy
            ptbox[:, 2] = preg[b, gy, gx][:, 2].sigmoid() * W
            ptbox[:, 3] = preg[b, gy, gx][:, 3].sigmoid() * H

            iou = self.bbox_iou(ptbox, gt_box[0])
            f = iou > iou.mean()
            b, gy, gx = b[f], gy[f], gx[f]

            iou = iou[f]
            iou_loss = (1.0 - iou).mean()

            # CrossEntropyLoss expects raw logits (no manual log needed)
            cls_loss = cls_criterion(pcls[b, gy, gx], gt_cls[0][f])

            # IoU-aware objectness target: high-IoU boxes get higher confidence targets
            tobj[b, gy, gx] = iou.float()
            n = torch.bincount(b)
            factor[b, gy, gx] = (1. / (n[b] / (H * W))) * 0.25

        obj_loss = (obj_criterion(pobj, tobj) * factor).mean()

        loss = (iou_loss * 8) + (obj_loss * 16) + cls_loss

        return iou_loss, obj_loss, cls_loss, loss
