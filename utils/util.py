import warnings

warnings.filterwarnings("ignore")

import copy
import math
import random

import numpy
import torch


def setup_seed():
    """
    Setup random seed.
    """
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_multi_processes():
    """
    Setup multi-processing environment variables.
    """
    import cv2
    from os import environ
    from platform import system

    # set multiprocess start method as `fork` to speed up the training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # disable opencv multithreading to avoid system being overloaded
    cv2.setNumThreads(0)

    # setup OMP threads
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = '1'

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = '1'


def make_anchors(x, strides, offset=0.5):
    assert x is not None
    anchor_tensor, stride_tensor = [], []
    dtype, device = x[0].dtype, x[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_tensor), torch.cat(stride_tensor)


def compute_metric(output, target, iou_v):
    # intersection(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2) = target[:, 1:].unsqueeze(1).chunk(2, 2)
    (b1, b2) = output[:, :4].unsqueeze(0).chunk(2, 2)
    intersection = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    # IoU = intersection / (area1 + area2 - intersection)
    iou = intersection / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - intersection + 1e-7)

    correct = numpy.zeros((output.shape[0], iou_v.shape[0]))
    correct = correct.astype(bool)
    for i in range(len(iou_v)):
        # IoU > threshold and classes match
        x = torch.where((iou >= iou_v[i]) & (target[:, 0:1] == output[:, 5]))
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1),
                                 iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[numpy.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[numpy.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=output.device)


def non_max_suppression(outputs, confidence_threshold=0.001):
    return [i[i[:, 4] > confidence_threshold] for i in outputs]


def smooth(y, f=0.1):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = numpy.ones(nf // 2)  # ones padding
    yp = numpy.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return numpy.convolve(yp, numpy.ones(nf) / nf, mode='valid')  # y-smoothed


def plot_pr_curve(px, py, ap, names, save_dir):
    from matplotlib import pyplot
    fig, ax = pyplot.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = numpy.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label="all classes %.3f mAP@0.5" % ap[:, 0].mean())
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    pyplot.close(fig)


def plot_curve(px, py, names, save_dir, x_label="Confidence", y_label="Metric"):
    from matplotlib import pyplot

    figure, ax = pyplot.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)

    y = smooth(py.mean(0), f=0.05)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.3f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{y_label}-Confidence Curve")
    figure.savefig(save_dir, dpi=250)
    pyplot.close(figure)


def compute_ap(tp, conf, output, target, plot=False, names=(), eps=1E-16):
    """
    Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Object-ness value from 0-1 (nparray).
        output:  Predicted object classes (nparray).
        target:  True object classes (nparray).
    # Returns
        The average precision
    """
    # Sort by object-ness
    i = numpy.argsort(-conf)
    tp, conf, output = tp[i], conf[i], output[i]

    # Find unique classes
    unique_classes, nt = numpy.unique(target, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    p = numpy.zeros((nc, 1000))
    r = numpy.zeros((nc, 1000))
    ap = numpy.zeros((nc, tp.shape[1]))
    px, py = numpy.linspace(start=0, stop=1, num=1000), []  # for plotting
    for ci, c in enumerate(unique_classes):
        i = output == c
        nl = nt[ci]  # number of labels
        no = i.sum()  # type: ignore number of outputs
        if no == 0 or nl == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (nl + eps)  # recall curve
        # negative x, xp because xp decreases
        r[ci] = numpy.interp(-px, -conf[i], recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = numpy.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            m_rec = numpy.concatenate(([0.0], recall[:, j], [1.0]))
            m_pre = numpy.concatenate(([1.0], precision[:, j], [0.0]))

            # Compute the precision envelope
            m_pre = numpy.flip(numpy.maximum.accumulate(numpy.flip(m_pre)))

            # Integrate area under curve
            x = numpy.linspace(start=0, stop=1, num=101)  # 101-point interp (COCO)
            ap[ci, j] = numpy.trapezoid(numpy.interp(x, m_rec, m_pre), x)  # integrate
            if plot and j == 0:
                py.append(numpy.interp(px, m_rec, m_pre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    if plot:
        names = dict(enumerate(names))  # to dict
        names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
        plot_pr_curve(px, py, ap, names, save_dir="./weights/PR_curve.png")
        plot_curve(px, f1, names, save_dir="./weights/F1_curve.png", y_label="F1")
        plot_curve(px, p, names, save_dir="./weights/P_curve.png", y_label="Precision")
        plot_curve(px, r, names, save_dir="./weights/R_curve.png", y_label="Recall")
    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    m_pre, m_rec = p.mean(), r.mean()
    map50, mean_ap = ap50.mean(), ap.mean()
    return tp, fp, m_pre, m_rec, f1.mean(), map50, mean_ap


def strip_optimizer(filename):
    x = torch.load(filename, map_location="cpu", weights_only=False)
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, f=filename)


def clip_gradients(model, max_norm=10.0):
    parameters = model.parameters()
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)


def load_weight(model, ckpt):
    dst = model.state_dict()
    src = torch.load(ckpt, weights_only=False)['model'].float().cpu()

    ckpt = {}
    for k, v in src.state_dict().items():
        if k in dst and v.shape == dst[k].shape:
            ckpt[k] = v

    model.load_state_dict(state_dict=ckpt, strict=False)
    return model


def set_params(model, decay):
    # norm layer types from torch.nn (e.g. BatchNorm, LayerNorm)
    norm_types = tuple(m for n, m in torch.nn.__dict__.items()
                       if "Norm" in n and isinstance(m, type))
    p4 = []
    p1 = []
    p2 = []
    p3 = []

    for m in model.modules():
        for name, param in m.named_parameters(recurse=False):

            if param.ndim >= 2:
                p4.append(param)
            elif name.endswith(".bias"):
                p3.append(param)
            elif isinstance(m, norm_types):
                p2.append(param)
            else:
                p1.append(param)

    groups = [{"params": p4, "weight_decay": decay, "param_group": "muon", },
              {"params": p1, "weight_decay": decay, "param_group": "weight"},
              {"params": p2, "weight_decay": 0.0, "param_group": "norm"},
              {"params": p3, "weight_decay": 0.0, "param_group": "bias"}]

    return groups


def plot_lr(args, optimizer, scheduler, num_steps):
    from matplotlib import pyplot

    optimizer = copy.copy(optimizer)
    scheduler = copy.copy(scheduler)

    y = []
    for epoch in range(args.epochs):
        for i in range(num_steps):
            step = i + num_steps * epoch
            scheduler.step(step, optimizer)
            y.append(optimizer.param_groups[0]['lr'])
    pyplot.plot(y, '.-', label='LR')
    pyplot.xlabel('step')
    pyplot.ylabel('LR')
    pyplot.grid()
    pyplot.xlim(0, args.epochs * num_steps)
    pyplot.ylim(0)
    pyplot.savefig('./weights/lr.png', dpi=200)
    pyplot.close()


class CosineLR:
    def __init__(self, args, params, num_steps):
        max_lr = params['max_lr']
        min_lr = params['min_lr']

        warmup_steps = int(max(params['warmup_epochs'] * num_steps, 100))
        decay_steps = int(args.epochs * num_steps - warmup_steps)

        warmup_lr = numpy.linspace(min_lr, max_lr, int(warmup_steps))

        decay_lr = []
        for step in range(1, decay_steps + 1):
            alpha = math.cos(math.pi * step / decay_steps)
            decay_lr.append(min_lr + 0.5 * (max_lr - min_lr) * (1 + alpha))

        self.total_lr = numpy.concatenate((warmup_lr, decay_lr))

    def step(self, step, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.total_lr[step]


class LinearLR:
    def __init__(self, args, params, num_steps):
        max_lr = params['max_lr']
        min_lr = params['min_lr']

        warmup_steps = int(max(params['warmup_epochs'] * num_steps, 100))
        decay_steps = int(args.epochs * num_steps - warmup_steps)

        warmup_lr = numpy.linspace(min_lr, max_lr, int(warmup_steps), endpoint=False)
        decay_lr = numpy.linspace(max_lr, min_lr, decay_steps)

        self.total_lr = numpy.concatenate((warmup_lr, decay_lr))

    def step(self, step, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.total_lr[step]


class EMA:
    """
    Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = copy.deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        if hasattr(model, 'module'):
            model = model.module
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        if not math.isnan(float(v)):
            self.num = self.num + n
            self.sum = self.sum + v * n
            self.avg = self.sum / self.num


def compute_iou(box1, box2, eps=1e-7):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = ((b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) *
             (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0))

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
    c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4  # center dist**2
    v = (4 / math.pi ** 2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)  # CIoU


def wh2xy(x):
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height

    y = torch.empty_like(x, dtype=x.dtype)
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y


def xy2wh(x):
    x1 = x[..., 0]
    y1 = x[..., 1]
    x2 = x[..., 2]
    y2 = x[..., 3]

    y = torch.empty_like(x, dtype=x.dtype)
    y[..., 0] = (x1 + x2) / 2  # x center
    y[..., 1] = (y1 + y2) / 2  # y center
    y[..., 2] = x2 - x1  # width
    y[..., 3] = y2 - y1  # height
    return y


class BoxLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(pred_dist, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum,
                fg_mask, size, stride):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = compute_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # L1 loss
        x1y1, x2y2 = target_bboxes.chunk(2, -1)
        target_ltrb = torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1)
        # normalize ltrb by image size
        target_ltrb = target_ltrb * stride
        target_ltrb[..., 0::2] /= size[1]
        target_ltrb[..., 1::2] /= size[0]
        pred_dist = pred_dist * stride
        pred_dist[..., 0::2] /= size[1]
        pred_dist[..., 1::2] /= size[0]
        loss_l1 = torch.nn.functional.l1_loss(pred_dist[fg_mask],
                                              target_ltrb[fg_mask],
                                              reduction="none").mean(-1, keepdim=True) * weight
        loss_l1 = loss_l1.sum() / target_scores_sum

        return loss_iou, loss_l1


class Assigner(torch.nn.Module):
    def __init__(self, top_k1, top_k2, nc, stride):
        super().__init__()
        self.nc = nc
        self.top_k1 = top_k1
        self.top_k2 = top_k2 or top_k1
        self.stride = stride
        self.alpha = 0.5
        self.beta = 6
        self.eps = 1e-9

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        batch_size = pd_scores.shape[0]
        n_max_boxes = gt_bboxes.shape[1]

        if n_max_boxes == 0:
            return (torch.zeros_like(pd_bboxes),
                    torch.zeros_like(pd_scores),
                    torch.zeros_like(pd_scores[..., 0]),
                    torch.zeros_like(pd_scores[..., 0]))

        gt_bboxes_xywh = xy2wh(gt_bboxes)
        wh_mask = gt_bboxes_xywh[..., 2:] < self.stride[0]  # the smallest stride
        stride_val = torch.tensor(self.stride[1], dtype=gt_bboxes_xywh.dtype, device=gt_bboxes_xywh.device)
        gt_bboxes_xywh[..., 2:] = torch.where((wh_mask * mask_gt).bool(), stride_val, gt_bboxes_xywh[..., 2:])
        gt_bboxes_xy = wh2xy(gt_bboxes_xywh)

        n_anchors = anc_points.shape[0]
        bs, n_boxes, _ = gt_bboxes_xy.shape
        lt, rb = gt_bboxes_xy.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((anc_points[None] - lt, rb - anc_points[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        mask_in_gts = bbox_deltas.amin(3).gt_(1e-9)

        na = pd_bboxes.shape[-2]
        mask = (mask_in_gts * mask_gt).bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([batch_size, n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([batch_size, n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, batch_size, n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=batch_size).view(-1, 1).expand(-1, n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        bbox_scores[mask] = pd_scores[ind[0], :, ind[1]][mask]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, n_max_boxes, -1, -1)[mask]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask]
        overlaps[mask] = compute_iou(gt_boxes, pd_boxes).squeeze(-1).clamp_(0)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)

        # Get topk_metric mask, (b, max_num_obj, h*w)
        top_k_mask = mask_gt.expand(-1, -1, self.top_k1).bool()
        # (b, max_num_obj, topk)
        top_k_metrics, top_k_indices = torch.topk(align_metric, self.top_k1, dim=-1, largest=True)
        if top_k_mask is None:
            top_k_mask = (top_k_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(top_k_indices)
        # (b, max_num_obj, topk)
        top_k_indices.masked_fill_(~top_k_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(align_metric.shape, dtype=torch.int8, device=top_k_indices.device)
        ones = torch.ones_like(top_k_indices[:, :, :1], dtype=torch.int8, device=top_k_indices.device)
        for k in range(self.top_k1):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, top_k_indices[:, :, k: k + 1], ones)
        # Filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)

        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = count_tensor.to(align_metric.dtype) * mask_in_gts * mask_gt

        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)

            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            fg_mask = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float().sum(-2)

        if self.top_k2 != self.top_k1:
            max_overlaps_idx = torch.topk(align_metric * mask_pos, self.top_k2, dim=-1, largest=True).indices
            topk_idx = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)  # update mask_pos
            topk_idx.scatter_(-1, max_overlaps_idx, 1.0)
            fg_mask = (mask_pos * topk_idx).sum(-2)
        # Find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)

        # Assigned target
        batch_ind = torch.arange(end=batch_size, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_idx = target_gt_idx + batch_ind * n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_idx]

        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        target_scores = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                    dtype=torch.int64, device=target_labels.device)  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_bboxes, target_scores, fg_mask.bool(), target_gt_idx


class Loss:
    def __init__(self, model, params, top_k1=10, top_k2=None):

        self.nc = model.head.nc
        self.no = model.head.nc + 4
        self.stride = model.head.stride

        self.params = params
        self.device = next(model.parameters()).device

        self.box_loss = BoxLoss().to(self.device)
        self.cls_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.assigner = Assigner(top_k1, top_k2, self.nc, self.stride.tolist())

    def __call__(self, outputs, targets):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (outputs["boxes"].permute(0, 2, 1).contiguous(),
                                    outputs["scores"].permute(0, 2, 1).contiguous())
        anchor_points, stride_tensor = make_anchors(outputs["x"], self.stride)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]

        size = torch.tensor(outputs["x"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((targets["idx"].view(-1, 1), targets["cls"].view(-1, 1), targets["box"]), 1)
        targets = targets.to(self.device)

        nl, ne = targets.shape
        if nl == 0:
            y = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            y = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    y[j, :n] = targets[matches, 1:]
            y[..., 1:5] = wh2xy(y[..., 1:5].mul_(size[[1, 0, 1, 0]]))

        gt_labels, gt_bboxes = y.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        lt, rb = pred_distri.chunk(2, -1)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        pred_bboxes = torch.cat((x1y1, x2y2), -1)  # xyxy bbox

        targets = self.assigner(pred_scores.detach().sigmoid(),
                                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                                anchor_points * stride_tensor,
                                gt_labels, gt_bboxes, mask_gt)
        target_bboxes, target_scores, fg_mask, target_gt_idx = targets

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        loss[1] = self.cls_loss(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.box_loss(pred_distri, pred_bboxes, anchor_points,
                                             target_bboxes / stride_tensor, target_scores, target_scores_sum,
                                             fg_mask, size, stride_tensor)

        loss[0] *= self.params['box']  # box gain
        loss[1] *= self.params['cls']  # cls gain
        loss[2] *= self.params['dfl']  # dfl gain

        return loss


class ComputeLoss:
    def __init__(self, args, params, model):
        self.args = args
        self.params = params

        if hasattr(model, 'module'):
            model = model.module

        self.loss1 = Loss(model, self.params, top_k1=10)
        self.loss2 = Loss(model, self.params, top_k1=7, top_k2=1)

        self.total = 1.0
        self.updates = 0

        # init gain
        self.a = 0.8
        self.b = self.a
        self.c = self.total - self.a

        # final gain
        self.d = 0.1

    def __call__(self, outputs, targets):
        loss1 = self.loss1(outputs[0], targets)
        loss2 = self.loss2(outputs[1], targets)

        loss = self.a * loss1 + self.c * loss2
        return loss[0], loss[1], loss[2]

    def update(self) -> None:
        self.updates += 1
        self.a = self.decay(self.updates)
        self.c = max(self.total - self.a, 0)

    def decay(self, x) -> float:
        return max(1 - x / max(self.args.epochs - 1, 1), 0) * (self.b - self.d) + self.d


def orthogonalize(grad, eps=1e-7):
    """
    Newton-Schulz orthogonalization
    """
    if grad.ndim != 2:
        raise ValueError(f"Expected a 2-D tensor, got shape {grad.shape}")

    x = grad.bfloat16()
    x = x / (x.norm() + eps)  # scale so top singular value ≤ 1

    transposed = grad.size(0) > grad.size(1)
    if transposed:
        x = x.T  # work in wide form (rows ≤ cols)

    for a, b, c in [(3.4445, -4.7750, 2.0315), ] * 5:
        A = x @ x.T
        x = a * x + (b * A + c * (A @ A)) @ x

    if transposed:
        x = x.T

    return x


def muon_update(grad, momentum_buf, beta):
    momentum_buf.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum_buf, beta)

    # Flatten conv filters from (out, in, kH, kW) → (out, in·kH·kW)
    if update.ndim == 4:
        update = update.view(update.size(0), -1)

    update = orthogonalize(update)

    # Scale so that wider matrices don't shrink the effective update norm
    aspect_scale = max(1.0, grad.size(-2) / grad.size(-1)) ** 0.5
    return update * aspect_scale


class MuSGD(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, muon=0.1, sgd=1.0):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
        self.muon = muon
        self.sgd = sgd

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if group["param_group"] == "muon":
                    self._muon_step(p, grad, state, lr, momentum, weight_decay)
                else:
                    self._sgd_step(p, grad, state, lr, momentum, weight_decay)

        return loss

    def _init_muon_state(self, p):
        self.state[p]["momentum_buffer_muon"] = torch.zeros_like(p)
        self.state[p]["momentum_buffer_sgd"] = torch.zeros_like(p)

    def _init_sgd_state(self, p):
        self.state[p]["momentum_buffer"] = torch.zeros_like(p)

    def _muon_step(self, p, grad, state, lr, momentum, weight_decay):
        if not state:
            self._init_muon_state(p)

        # --- Muon component ---
        muon_update_val = muon_update(grad, state["momentum_buffer_muon"], momentum)
        p.add_(muon_update_val.reshape(p.shape), alpha=-(lr * self.muon))

        # --- SGD component (with optional weight decay) ---
        grad_wd = grad.add(p, alpha=weight_decay) if weight_decay != 0 else grad

        buf = state["momentum_buffer_sgd"]
        buf.mul_(momentum).add_(grad_wd)
        sgd_update = grad_wd.add(buf, alpha=momentum)
        p.add_(sgd_update, alpha=-(lr * self.sgd))

    def _sgd_step(self, p, grad, state, lr, momentum, weight_decay):
        if not state:
            self._init_sgd_state(p)

        grad_wd = grad.add(p, alpha=weight_decay) if weight_decay != 0 else grad

        buf = state["momentum_buffer"]
        buf.mul_(momentum).add_(grad_wd)
        update = grad_wd.add(buf, alpha=momentum)
        p.add_(update, alpha=-lr)
