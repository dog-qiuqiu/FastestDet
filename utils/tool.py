import yaml
import torch
import torchvision


class LoadYaml:
    def __init__(self, path):
        with open(path, encoding='utf8') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        self.val_txt = data["DATASET"]["VAL"]
        self.train_txt = data["DATASET"]["TRAIN"]
        self.names = data["DATASET"]["NAMES"]

        self.learn_rate = data["TRAIN"]["LR"]
        self.batch_size = data["TRAIN"]["BATCH_SIZE"]
        self.milestones = data["TRAIN"]["MILESTIONES"]
        self.end_epoch = data["TRAIN"]["END_EPOCH"]
        self.num_workers = data["TRAIN"].get("NUM_WORKERS", 4)

        # Training enhancements (all have safe defaults for backwards compatibility)
        self.warmup_epochs = data["TRAIN"].get("WARMUP_EPOCHS", 5)
        self.amp = data["TRAIN"].get("AMP", False)
        self.grad_clip = data["TRAIN"].get("GRAD_CLIP", 10.0)
        self.scheduler = data["TRAIN"].get("SCHEDULER", "multistep")

        self.input_width = data["MODEL"]["INPUT_WIDTH"]
        self.input_height = data["MODEL"]["INPUT_HEIGHT"]

        self.category_num = data["MODEL"]["NC"]

        print("Load yaml success...")

    def __repr__(self):
        return (
            f"Dataset: train={self.train_txt}, val={self.val_txt}\n"
            f"Model: {self.input_width}x{self.input_height}, classes={self.category_num}\n"
            f"Train: lr={self.learn_rate}, batch={self.batch_size}, epochs={self.end_epoch}, "
            f"warmup={self.warmup_epochs}, amp={self.amp}, grad_clip={self.grad_clip}, "
            f"scheduler={self.scheduler}"
        )


def handle_preds(preds, device, conf_thresh=0.25, nms_thresh=0.45):
    total_bboxes, output_bboxes = [], []
    N, C, H, W = preds.shape
    bboxes = torch.zeros((N, H, W, 6))
    pred = preds.permute(0, 2, 3, 1)

    pobj = pred[:, :, :, 0].unsqueeze(dim=-1)
    preg = pred[:, :, :, 1:5]
    pcls = pred[:, :, :, 5:]

    # Confidence = obj^0.6 * cls^0.4: geometric mean that balances localisation
    # confidence (objectness) and recognition confidence (class score).
    # The asymmetric exponents down-weight classification to reduce false negatives
    # on classes that are hard to distinguish but easy to locate.
    bboxes[..., 4] = (pobj.squeeze(-1) ** 0.6) * (pcls.max(dim=-1)[0] ** 0.4)
    bboxes[..., 5] = pcls.argmax(dim=-1)

    gy, gx = torch.meshgrid([torch.arange(H), torch.arange(W)])
    bw, bh = preg[..., 2].sigmoid(), preg[..., 3].sigmoid()
    bcx = (preg[..., 0].tanh() + gx.to(device)) / W
    bcy = (preg[..., 1].tanh() + gy.to(device)) / H

    x1, y1 = bcx - 0.5 * bw, bcy - 0.5 * bh
    x2, y2 = bcx + 0.5 * bw, bcy + 0.5 * bh

    bboxes[..., 0], bboxes[..., 1] = x1, y1
    bboxes[..., 2], bboxes[..., 3] = x2, y2
    bboxes = bboxes.reshape(N, H * W, 6)
    total_bboxes.append(bboxes)

    batch_bboxes = torch.cat(total_bboxes, 1)

    for p in batch_bboxes:
        output, temp = [], []
        b, s, c = [], [], []
        t = p[:, 4] > conf_thresh
        pb = p[t]
        for bbox in pb:
            obj_score = bbox[4]
            category = bbox[5]
            x1, y1 = bbox[0], bbox[1]
            x2, y2 = bbox[2], bbox[3]
            s.append([obj_score])
            c.append([category])
            b.append([x1, y1, x2, y2])
            temp.append([x1, y1, x2, y2, obj_score, category])
        if len(b) > 0:
            b = torch.Tensor(b).to(device)
            c = torch.Tensor(c).squeeze(1).to(device)
            s = torch.Tensor(s).squeeze(1).to(device)
            keep = torchvision.ops.batched_nms(b, s, c, nms_thresh)
            for i in keep:
                output.append(temp[i])
        output_bboxes.append(torch.Tensor(output))
    return output_bboxes
