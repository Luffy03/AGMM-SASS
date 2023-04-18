import numpy as np
import logging
import os
import torch
import torch.nn.functional as F
from math import *
from PIL import Image
import matplotlib.pyplot as plt


def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255,  0,  0])
        cmap[13] = np.array([0,  0, 142])
        cmap[14] = np.array([0,  0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0,  0, 230])
        cmap[18] = np.array([119, 11, 32])

        cmap[19] = np.array([0, 0, 0])
        cmap[255] = np.array([0, 0, 0])

    return cmap


class meanIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        return iu, np.nanmean(iu)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[2]
    cols_missing = target_size[1] - img.shape[3]
    padded_img = F.pad(img, (0, 0, rows_missing, cols_missing), 'constant', 0)
    return padded_img


def pre_slide(model, image, num_classes=21, tile_size=(321, 321), tta=False):
    image_size = image.shape  # bigger than (1, 3, 512, 512), i.e. (1,3,1024,1024)
    overlap = 2 / 3  # 每次滑动的重合率为1/2

    stride = ceil(tile_size[0] * (1 - overlap))  # 滑动步长:769*(1-1/3) = 513
    tile_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)  # 行滑动步数:(1024-769)/513 + 1 = 2
    tile_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)  # 列滑动步数:(2048-769)/513 + 1 = 4

    full_probs = torch.zeros((1, num_classes, image_size[2], image_size[3])).cuda()  # 初始化全概率矩阵 shape(1024,2048,19)

    count_predictions = torch.zeros((1, 1, image_size[2], image_size[3])).cuda()  # 初始化计数矩阵 shape(1024,2048,19)
    tile_counter = 0  # 滑动计数0

    for row in range(tile_rows):  # row = 0,1
        for col in range(tile_cols):  # col = 0,1,2,3
            x1 = int(col * stride)  # 起始位置x1 = 0 * 513 = 0
            y1 = int(row * stride)  # y1 = 0 * 513 = 0
            x2 = min(x1 + tile_size[1], image_size[3])  # 末位置x2 = min(0+769, 2048)
            y2 = min(y1 + tile_size[0], image_size[2])  # y2 = min(0+769, 1024)
            x1 = max(int(x2 - tile_size[1]), 0)  # 重新校准起始位置x1 = max(769-769, 0)
            y1 = max(int(y2 - tile_size[0]), 0)  # y1 = max(769-769, 0)

            img = image[:, :, y1:y2, x1:x2]  # 滑动窗口对应的图像 imge[:, :, 0:769, 0:769]
            padded_img = pad_image(img, tile_size)  # padding 确保扣下来的图像为769*769

            tile_counter += 1  # 计数加1
            # print("Predicting tile %i" % tile_counter)

            # 将扣下来的部分传入网络，网络输出概率图。
            # use softmax
            if tta:
                padded = model(padded_img, True)
            else:
                padded = model(padded_img)[0] if isinstance(model(img), tuple) else model(padded_img)
                padded = F.softmax(padded, dim=1)

            pre = padded[:, :, 0:img.shape[2], 0:img.shape[3]]  # 扣下相应面积 shape(769,769,19)
            count_predictions[:, :, y1:y2, x1:x2] += 1  # 窗口区域内的计数矩阵加1
            full_probs[:, :, y1:y2, x1:x2] += pre  # 窗口区域内的全概率矩阵叠加预测结果

    # average the predictions in the overlapping regions
    full_probs /= count_predictions  # 全概率矩阵 除以 计数矩阵 即得 平均概率

    return full_probs   # 返回整张图的平均概率 shape(1, 1, 1024,2048)


def ms_test(model, img):
    n, c, h, w = img.size()
    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    final_result = None

    for scale in scales:
        cur_h, cur_w = int(h * scale), int(w * scale)
        cur_x = F.interpolate(img, size=(cur_h, cur_w), mode='bilinear', align_corners=True)

        out = model(cur_x)
        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
        final_result = out if final_result is None else (final_result + out)

        out = model(cur_x.flip(3)).flip(3)
        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
        final_result += out

    return final_result / 14


def evaluate(model, loader, mode, cfg):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    metric = meanIOU(num_classes=cfg['nclass'])

    with torch.no_grad():
        for img, mask, id in loader:
            img = img.cuda()

            if mode == 'sliding_window':
                final = pre_slide(model, img, num_classes=cfg['nclass'],
                                 tile_size=(cfg['crop_size'], cfg['crop_size']), tta=False)

                pred = final.argmax(dim=1)

            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]

                oh, ow = img.size()[-2:]
                input = F.interpolate(img, size=(cfg['crop_size'], cfg['crop_size']), mode='bilinear')
                pred = model(input)
                pred = F.interpolate(pred, size=(oh, ow), mode='bilinear').argmax(dim=1)

            metric.add_batch(pred.cpu().numpy(), mask.numpy())

    iou_class, mIOU = metric.evaluate()
    mIOU = mIOU * 100.0

    return mIOU, iou_class


if __name__ == '__main__':
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0


    r = g = b = 0
    c = 15

    for i in range(256):
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3
    print(r, g, b)
