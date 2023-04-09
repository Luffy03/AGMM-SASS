import argparse
from itertools import cycle
import logging
import os
import pprint
import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
import yaml

from dataset.sass import *
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log, evaluate
from util.dist_helper import setup_distributed

os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2"
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '28890'
# sh tools/train_voc.sh 3 28890

parser = argparse.ArgumentParser(description='Sparsely-annotated Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def main():
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    rank, word_size = setup_distributed(port=args.port)

    if rank == 0:
        logger.info('{}\n'.format(pprint.pformat(cfg)))

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = DeepLabV3Plus(cfg, aux=cfg['aux'])

    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=False)

    ohem = False if cfg['criterion']['name'] == 'CELoss' else True
    use_weight = True if cfg['dataset'] == 'cityscapes' else False

    if cfg['dataset'] == 'pascal':
        trainset = VocDataset(cfg['dataset'], cfg['data_root'], cfg['mode'],
                              cfg['crop_size'], cfg['aug'])
        valset = VocDataset(cfg['dataset'], cfg['data_root'], 'val', None)

    else:
        trainset = CityDataset(cfg['dataset'], cfg['data_root'], cfg['mode'],
                               cfg['crop_size'], cfg['aug'])
        valset = CityDataset(cfg['dataset'], cfg['data_root'], 'val', None)

    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],
                             pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=2,
                           drop_last=False, sampler=valsampler)

    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    previous_best = 0.0

    for epoch in range(cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.4f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        model.train()
        loss_m = AverageMeter()
        seg_m = AverageMeter()
        gmm_m = AverageMeter()

        trainsampler.set_epoch(epoch)

        for i, (img, mask, cls_label, id) in enumerate(trainloader):
            img, mask, cls_label = img.cuda(), mask.cuda(), cls_label.cuda()
            feat, pred = model(img)

            seg_loss = loss_calc(pred, mask,
                                 ignore_index=cfg['nclass'], multi=False,
                                 class_weight=use_weight, ohem=ohem)

            # Gaussian
            cur_cls_label = build_cur_cls_label(mask, cfg['nclass'])
            pred_cl = clean_mask(pred, cls_label, True)
            vecs, proto_loss = cal_protypes(feat, mask, cfg['nclass'])
            res = GMM(feat, vecs, pred_cl, mask, cur_cls_label)
            gmm_loss = cal_gmm_loss(pred.softmax(1), res, cur_cls_label, mask) + proto_loss

            # total loss
            loss = seg_loss + gmm_loss
            # for ablation
            # loss = seg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_m.update(loss.item(), img.size()[0])
            seg_m.update(seg_loss.item(), img.size()[0])
            gmm_m.update(gmm_loss.item(), img.size()[0])

            iters += 1
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            if (i % (max(2, len(trainloader) // 8)) == 0) and (rank == 0):
                logger.info('Iters:{:}, loss:{:.3f}, seg_loss:{:.3f}, '
                            'gmm_loss:{:.3f}'.format
                            (i, loss_m.avg, seg_m.avg, gmm_m.avg))

        if cfg['dataset'] == 'cityscapes':
            eval_mode = 'center_crop' if epoch < cfg['epochs'] - 20 else 'sliding_window'
        else:
            eval_mode = 'original'
        mIOU, iou_class = evaluate(model, valloader, eval_mode, cfg)

        if rank == 0:
            logger.info('***** Evaluation {} ***** >>>> meanIOU: {:.2f}\n'.format(eval_mode, mIOU))

        if mIOU > previous_best and rank == 0:
            if previous_best != 0:
                os.remove(os.path.join(args.save_path, '%s_%.2f.pth' % (cfg['backbone'], previous_best)))
            previous_best = mIOU
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_path, '%s_%.2f.pth' % (cfg['backbone'], mIOU)))


if __name__ == '__main__':
    main()
