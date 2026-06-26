import warnings

warnings.filterwarnings("ignore")

import os
import csv
import copy
import argparse
import datetime

import yaml
import thop
import tqdm
import torch

from nets import nn
from utils import util
from utils.dataset import Dataset

data_dir = '/home/jahongir/Projects/Dataset/COCO'


def train(args, params):
    # Model
    model = nn.yolo_v26_n(len(params['names']))
    model.cuda()

    # Optimizer
    args.accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params['weight_decay'] *= args.batch_size * args.world_size * args.accumulate / 64

    args.date = datetime.datetime.now()
    optimizer = util.MuSGD(util.set_params(model, params['weight_decay']),
                           lr=params['min_lr'], momentum=params['momentum'])

    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None

    filenames = []
    with open(f'{data_dir}/train2017.txt') as f:
        for filename in f.readlines():
            filename = filename.rstrip().split('/')[-1]
            filenames.append(f'{data_dir}/images/train2017/' + filename)

    sampler = None
    dataset = Dataset(filenames, args.input_size, params, augment=True)

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    loader = torch.utils.data.DataLoader(dataset,
                                         args.batch_size, sampler is None, sampler,
                                         num_workers=4, pin_memory=True, collate_fn=Dataset.collate_fn)

    if args.distributed:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    step = 0
    best = 0
    num_steps = len(loader)
    amp_scale = torch.amp.GradScaler()
    criterion = util.ComputeLoss(args, params, model)
    scheduler = util.LinearLR(args, params, num_steps)
    with open('weights/step.csv', 'w') as log:
        if args.local_rank == 0:
            logger = csv.DictWriter(log, fieldnames=['epoch',
                                                     'box', 'cls', 'dfl',
                                                     'Recall', 'Precision', 'F1', 'mAP@50', 'mAP'])
            logger.writeheader()

        for epoch in range(args.epochs):
            model.train()

            if args.distributed:
                sampler.set_epoch(epoch)
            if args.epochs - epoch == 10:
                loader.dataset.mosaic = False

            p_bar = loader

            if args.local_rank == 0:
                print(('\n' + '%10s' * 5) % ('epoch', 'memory', 'box', 'cls', 'dfl'))
                p_bar = tqdm.tqdm(p_bar, total=num_steps)

            optimizer.zero_grad()
            avg_box_loss = util.AverageMeter()
            avg_cls_loss = util.AverageMeter()
            avg_dfl_loss = util.AverageMeter()
            for samples, targets in p_bar:
                scheduler.step(step, optimizer)
                samples = samples.cuda().float() / 255

                # Forward
                with torch.amp.autocast('cuda'):
                    outputs = model(samples)  # forward
                    loss_box, loss_cls, loss_dfl = criterion(outputs, targets)

                avg_box_loss.update(loss_box.item(), samples.size(0))
                avg_cls_loss.update(loss_cls.item(), samples.size(0))
                avg_dfl_loss.update(loss_dfl.item(), samples.size(0))

                loss_box *= args.batch_size  # loss scaled by batch_size
                loss_cls *= args.batch_size  # loss scaled by batch_size
                loss_dfl *= args.batch_size  # loss scaled by batch_size
                loss_box *= args.world_size  # gradient averaged between devices in DDP mode
                loss_cls *= args.world_size  # gradient averaged between devices in DDP mode
                loss_dfl *= args.world_size  # gradient averaged between devices in DDP mode

                # Backward
                amp_scale.scale(loss_box + loss_cls + loss_dfl).backward()

                # Optimize
                if step % args.accumulate == 0:
                    amp_scale.unscale_(optimizer)  # unscale gradients
                    util.clip_gradients(model)  # clip gradients
                    amp_scale.step(optimizer)  # optimizer.step
                    amp_scale.update()
                    optimizer.zero_grad()

                    if ema:
                        ema.update(model)

                torch.cuda.synchronize()

                # Log
                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'  # (GB)
                    s = ('%10s' * 2 + '%10.3g' * 3) % (f'{epoch + 1}/{args.epochs}', memory,
                                                       avg_box_loss.avg, avg_cls_loss.avg, avg_dfl_loss.avg)
                    p_bar.set_description(s)

                step += 1

            if args.local_rank == 0:
                criterion.update()

            if args.local_rank == 0:
                # mAP
                last = test(args, params, ema.ema)

                logger.writerow({'epoch': str(epoch + 1).zfill(3),
                                 'box': str(f'{avg_box_loss.avg:.3f}'),
                                 'cls': str(f'{avg_cls_loss.avg:.3f}'),
                                 'dfl': str(f'{avg_dfl_loss.avg:.3f}'),
                                 'F1': str(f'{last[0]:.3f}'),
                                 'mAP': str(f'{last[1]:.3f}'),
                                 'mAP@50': str(f'{last[2]:.3f}'),
                                 'Recall': str(f'{last[3]:.3f}'),
                                 'Precision': str(f'{last[4]:.3f}')})
                log.flush()

                # Update best mAP
                if last[1] > best:
                    best = last[1]

                # Save model
                save = {'epoch': epoch + 1,
                        'model': copy.deepcopy(ema.ema),
                        'args': args,
                        'params': params}

                # Save last, best and delete
                torch.save(save, f='./weights/last.pt')
                if best == last[1]:
                    torch.save(save, f='./weights/best.pt')
                del save

    if args.local_rank == 0:
        util.strip_optimizer('./weights/best.pt')  # strip optimizers
        util.strip_optimizer('./weights/last.pt')  # strip optimizers


def test(args, params, model=None):
    filenames = []
    with open(f'{data_dir}/val2017.txt') as f:
        for filename in f.readlines():
            filename = filename.rstrip().split('/')[-1]
            filenames.append(f'{data_dir}/images/val2017/' + filename)

    dataset = Dataset(filenames, args.input_size, params, augment=False)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=4,
                                         shuffle=False, num_workers=1,
                                         pin_memory=True, collate_fn=Dataset.collate_fn)

    plot = False
    if not model:
        plot = True
        model = torch.load(f='./weights/best.pt', map_location='cuda', weights_only=False)
        print(model['epoch'])
        model = model['model'].float().fuse()

    model.half()
    model.eval()

    # Configure
    iou_v = torch.linspace(start=0.5, end=0.95, steps=10).cuda()  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    m_pre = 0
    m_rec = 0
    map50 = 0
    h_mean = 0
    mean_ap = 0
    metrics = []
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 5) % ('precision', 'recall', 'F1', 'mAP50', 'mAP'))
    for samples, targets in p_bar:
        samples = samples.cuda()
        samples = samples.half()  # uint8 to fp16/32
        samples = samples / 255.  # 0 - 255 to 0.0 - 1.0
        _, _, h, w = samples.shape  # batch-size, channels, height, width
        scale = torch.tensor((w, h, w, h)).cuda()
        # Inference
        with torch.no_grad():
            outputs = model(samples)
        # NMS
        outputs = util.non_max_suppression(outputs)
        # Metrics
        for i, output in enumerate(outputs):
            idx = targets['idx'] == i
            cls = targets['cls'][idx]
            box = targets['box'][idx]

            cls = cls.cuda()
            box = box.cuda()

            metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

            if output.shape[0] == 0:
                if cls.shape[0]:
                    metrics.append((metric, *torch.zeros((2, 0)).cuda(), cls.squeeze(-1)))
                continue
            # Evaluate
            if cls.shape[0]:
                target = torch.cat(tensors=(cls, util.wh2xy(box) * scale), dim=1)
                metric = util.compute_metric(output[:, :6], target, iou_v)
            # Append
            metrics.append((metric, output[:, 4], output[:, 5], cls.squeeze(-1)))

    # Compute metrics
    metrics = [torch.cat(x, dim=0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, h_mean, map50, mean_ap = util.compute_ap(*metrics,
                                                                       plot=plot,
                                                                       names=params["names"])
    # Print results
    print(('%10.3g' * 5) % (m_pre, m_rec, h_mean, map50, mean_ap))
    # Return results
    model.float()  # for training
    return h_mean, mean_ap, map50, m_rec, m_pre


def profile(args, params):
    shape = (1, 3, args.input_size, args.input_size)
    model = nn.yolo_v26_n(len(params['names'])).fuse()

    model.eval()
    model(torch.zeros(shape))

    x = torch.empty(shape)
    flops, num_params = thop.profile(model, inputs=[x], verbose=False)
    flops, num_params = thop.clever_format(nums=[2 * flops, num_params], format="%.3f")

    print(f'Number of parameters: {num_params}')
    print(f'Number of FLOPs: {flops}')

    if args.benchmark:
        # Latency
        model = nn.yolo_v26_n(len(params['names'])).fuse()
        model.eval()
        model.cuda()

        x = torch.zeros(shape).cuda()
        for i in range(10):
            model(x)
        total = 0
        import time
        for i in range(1_000):
            start = time.perf_counter()
            with torch.no_grad():
                model(x)
            total += time.perf_counter() - start

        print(f"Latency: {total / 1_000 * 1_000:.3f} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--epochs', default=600, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', default=True, action='store_true')

    args = parser.parse_args()

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    with open('utils/args.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)

    util.setup_seed()
    util.setup_multi_processes()

    if args.local_rank == 0:
        profile(args, params)

    if args.train:
        train(args, params)
    if args.test:
        test(args, params)

    # Clean
    if args.distributed:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
