from __future__ import division
import sys
import os
import math
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch
import random
import numpy as np
from torch.utils import data

from utils.tools import backup
from utils.logger import get_logger
from utils.utils import clip_gradient, PolyLr

from utils.dataloader import get_random_video_dataset
from utils.eval import eval_seg_sqc as eval1
from all_config.config_main import config
from model.TCCNet import TCCNet


def train(model, snapshot_path, device, writer):
    batch_size = config.video_batchsize
    base_lr = config.base_lr
    lr = base_lr

    if config.train_mode is 'main_training':
        label_data, test_data = get_random_video_dataset(config)
    else:
        print("all_config.train_mode_{} is invalid!".format(config.train_mode))
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    description = f'''{config.name}_{config.repo_name} starting training:
                                        model:          {config.model}
                                        train mode:     {config.train_mode}
                                        train sqc:      {len(label_data)}
                                        test sqc:       {len(test_data)}
                                        batch_sz:       {batch_size}
                                        base_lr:        {base_lr}
                                        '''
    logging.info(description)
    print(description)
    config.niters_per_epoch = int(math.ceil(len(label_data) * 1.0 // batch_size))

    # data loader
    train_loader = data.DataLoader(label_data,
                                   batch_size=batch_size,
                                   num_workers=8,
                                   shuffle=True,
                                   drop_last=True)
    test_loader = data.DataLoader(test_data,
                                  batch_size=batch_size,
                                  num_workers=8,
                                  shuffle=False)

    # optimizer
    optimizer = model.get_optimizer(base_lr)
    scheduler = PolyLr(optimizer, gamma=config.gamma,
                       minimum_lr=config.min_learning_rate,
                       max_iteration=len(train_loader) * config.nepochs,
                       warmup_iteration=config.warmup_iteration)

    model.train()
    best_metric = 0
    test_iou = 0
    train_iou = 0
    best_epoch = 0

    for epoch in range(config.nepochs):
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)

        if epoch % 2 == 0:
            label_data.shufle_data()
            train_loader = data.DataLoader(label_data,
                                           batch_size=batch_size,
                                           num_workers=8,
                                           shuffle=True,
                                           drop_last=True)

        total_seg, total_cps, total_byol, total_loss,  = 0, 0, 0, 0

        dataloader = iter(train_loader)
        for idx in pbar:
            lr = scheduler.get_curlr()
            minibatch = dataloader.__next__()
            Fs = minibatch['img']  # b, t, c, h, w
            Ms = minibatch['mask']  # b, t, c, h, w
            Bs = minibatch['border']
            Fs = Fs.to(device=device, dtype=torch.float32)
            Ms = Ms.to(device=device, dtype=torch.float32)
            Bs = Bs.to(device=device, dtype=torch.float32)

            optimizer.zero_grad()
            loss_dic, pred_dic = model.train_step_maintraining(Fs, Ms, Bs, device, epoch=epoch, idx=idx)
            loss = loss_dic['total']

            loss.backward()
            clip_gradient(optimizer, config.clip)
            optimizer.step()
            scheduler.step()
            model.update_target()

            # eval
            current_idx = epoch * config.niters_per_epoch + idx
            if current_idx % (config.save_frequency * config.niters_per_epoch // 15) == 0:
                # train_loss, train_metrics = eval1(model, train_loader, device)
                test_loss, test_metrics = eval1(model, test_loader, device)
                test_iou = test_metrics['JAp']

                ifsave = False
                if test_iou > best_metric:
                    ifsave = True
                    best_metric = test_iou
                    best_epoch = epoch

                if ifsave or (epoch > 29 and epoch % 5 == 0 and idx == 10):
                    try:
                        os.mkdir(snapshot_path)
                    except OSError:
                        pass
                    model.save_checkpoint(snapshot_path, f'epoch{epoch}_IOU{test_iou}')

            total_loss += loss.item()
            total_cps += loss_dic['cps'].item()
            total_seg += loss_dic['seg'].item()
            total_byol += loss_dic['byol'].item()

            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' total={:.4f} seg={:.4f} cps={:.4f} cut={:.4f}'.format(
                            total_loss / (idx+1), total_seg / (idx+1), total_cps/(idx+1), total_byol/(idx+1))
            print_str += ' metric{:.3f}% < {:.3f}%-{}:'.format(test_iou * 100, best_metric * 100, best_epoch)
            print_str += 'train-{:.3f}%'.format(train_iou * 100)
            pbar.set_description(print_str, refresh=False)

        if (epoch > 10 and epoch % 2 == 0):
            try:
                os.mkdir(snapshot_path)
            except OSError:
                pass
            model.save_checkpoint(snapshot_path, f'epoch{epoch}_IOU{test_iou}')

        loss_str = 'epoch_{}, lr_{:.4e}, total_{:.3f}, seg_{:3f}, cps_{:.3f}, byol_{:.3f}'.format(
            epoch, lr, total_loss / len(pbar), total_seg / len(pbar), total_cps / len(pbar), total_byol / len(pbar)
        )
        loss_str += ' metric{:.3f}% < {:.3f}%-{}:'.format(test_iou * 100, best_metric * 100, best_epoch)
        loss_str += 'train-{:.3f}%'.format(train_iou * 100)
        logging.info(loss_str)
        # print(total_loss / len(pbar), total_seg / len(pbar), total_cps / len(pbar), total_byol / len(pbar))

        try:
            os.mkdir(snapshot_path)
        except OSError:
            pass
        model.save_checkpoint(snapshot_path, f'epoch{epoch}_IOU{test_iou}')

        loss_dict = {
            'total': total_loss / len(pbar),
            'seg': total_seg / len(pbar),
            'cps': total_cps / len(pbar),
            'byol': total_byol / len(pbar),
        }
        iou_dic = {
            "test": test_iou,
        }
        writer.add_scalars(repo_name + '/loss', loss_dict, epoch)
        writer.add_scalars(repo_name + '/iou', iou_dic, epoch)

    writer.close()


if __name__ == "__main__":
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    snapshot_path = config.snapshot_path
    visualize_path = config.visualize_path
    repo_name = config.repo_name

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    log_file = "/log.log"
    logging = get_logger(log_dir=snapshot_path, log_file=log_file)
    writer = SummaryWriter(config.writer_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    backup(config.backup, snapshot_path)

    model = TCCNet(config)
    model = model.to(device=device)

    if config.load:
        model.load_checkpoint(config.load, logging)

    try:
        train(model, snapshot_path, device, writer)
    except KeyboardInterrupt:
        model.save_checkpoint(snapshot_path, 'INTERRUPTED')
        # torch.save(model.state_dict(), snapshot_path + '/INTERRUPTED.pth')
        logging.info('Saved interrupt')
        writer.close()
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
