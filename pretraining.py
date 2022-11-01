from __future__ import division
import sys
import math
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils import data

from utils.tools import backup
from utils.logger import get_logger
from utils.utils import clip_gradient

from utils.dataloader import *
from utils.eval import eval_seg_sqc as eval1
from all_config.config_pretraining import config
from model.TCCNet import *


def train(model, snapshot_path, device, writer):
    batch_size = config.video_batchsize
    base_lr = config.base_lr
    lr = base_lr

    # data loader
    if config.train_mode is 'pretraining':
        label_data, test_data = get_pesudo_video_dataset(config)
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
    logging.info(config.setting)
    logging.info(description)
    print(config.setting, '\n', description)
    config.niters_per_epoch = int(math.ceil(len(label_data) * 1.0 // batch_size))

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

    model.train()
    best_metric = 0
    test_iou = 0
    train_iou = 0
    best_epoch = 0

    for epoch in range(config.nepochs):
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)
        total_seg, total_prop, total_loss, total_byol = 0, 0, 0, 0

        dataloader = iter(train_loader)
        for idx in pbar:
            minibatch = dataloader.__next__()

            Fs = minibatch['img']  # b, t, c, h, w
            Ms = minibatch['mask']  # b, t, c, h, w
            Bs = minibatch['border']
            Fs = Fs.to(device=device, dtype=torch.float32)
            Ms = Ms.to(device=device, dtype=torch.float32)
            Bs = Bs.to(device=device, dtype=torch.float32)

            optimizer.zero_grad()
            loss_dic, pred_dic = model.train_step_pretrain(Fs, Ms, Bs, device, epoch=epoch, idx=idx)
            loss = loss_dic['total']
            loss.backward()
            clip_gradient(optimizer, config.clip)
            optimizer.step()
            model.update_target()

            # eval
            current_idx = epoch * config.niters_per_epoch + idx
            if current_idx % (config.save_frequency * config.niters_per_epoch) == 0:
                train_loss, train_metrics = eval1(model, train_loader, device)
                test_loss, test_metrics = eval1(model, test_loader, device)
                test_iou = test_metrics['JAp']
                train_iou = train_metrics["JAp"]

                ifsave = False
                if test_iou > best_metric:
                    ifsave = True
                    best_metric = test_iou
                    best_epoch = epoch

                if ifsave:
                    try:
                        os.mkdir(snapshot_path)
                    except OSError:
                        pass
                    model.save_checkpoint(snapshot_path, f'epoch{epoch}_IOU{test_iou}')

            total_loss += loss.item()
            total_prop += loss_dic['prop'].item()
            total_seg += loss_dic['seg'].item()
            total_byol += loss_dic['byol'].item()

            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.1e' % lr \
                        + ' total={:.4f} seg={:.4f} prop={:.4f} cut={:.4f}'.format(
                            total_loss / (idx+1), total_seg / (idx+1), total_prop/(idx+1), total_byol/(idx+1))
            print_str += ' metric{:.3f}% < {:.3f}%-{}:'.format(test_iou * 100, best_metric * 100, best_epoch)
            print_str += 'train-{:.3f}%'.format(train_iou * 100)
            pbar.set_description(print_str, refresh=False)

        loss_str = 'epoch_{}, lr{}, total_{}, seg_{}, prop_{}, byol_{}'.format(
            epoch, lr, total_loss / len(pbar), total_seg / len(pbar), total_prop / len(pbar), total_byol / len(pbar)
        )
        loss_str += ' metric{:.3f}% < {:.3f}%-{}:'.format(test_iou * 100, best_metric * 100, best_epoch)
        loss_str += 'train-{:.3f}%'.format(train_iou * 100)
        logging.info(loss_str)

        loss_dict = {
            'total': total_loss / len(pbar),
            'seg': total_seg / len(pbar),
            'prop': total_prop / len(pbar),
            'byol': total_byol / len(pbar),
        }
        iou_dic = {
            "train": train_iou,
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

    # model = eval(config.model)(config)
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
