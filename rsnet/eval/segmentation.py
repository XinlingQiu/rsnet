import numpy as np

from rsnet.dataset import RasterSampleDataset


class ConfusionMatrix:
    def __init__(self, num_classes, ignore_index):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, gt):
        n = self.num_classes
        mask = gt != self.ignore_index
        pred = pred[mask]
        gt = gt[mask]

        mask = (gt >= 0) & (gt < n)
        inds = n * gt[mask].astype(np.int64) + pred[mask]
        self.mat += np.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.fill(0)

    def pixel_accuracy(self):
        h = self.mat
        acc = np.diag(h).sum() / h.sum()
        return acc

    def overall_acc(self):
        return np.diag(self.mat).sum() / self.mat.sum()

    def pixel_accuracy_classwise(self):
        h = self.mat
        acc = np.diag(h) / h.sum(axis=1)
        return acc

    def iou(self):
        h = self.mat
        iou = np.diag(h) / (h.sum(axis=1) + h.sum(axis=0) - np.diag(h))
        return iou

    def fw_iou(self):
        h = self.mat
        freq = h.sum(axis=1) / h.sum()
        iou = self.iou()
        fw_iou = (freq[freq > 0] * iou[freq > 0]).sum()
        return fw_iou

    def precision(self):
        h = self.mat
        tp = np.diag(h)
        prec = tp / h.sum(axis=0)
        return prec

    def recall(self):
        h = self.mat
        tp = np.diag(h)
        recall = tp / h.sum(axis=1)
        return recall

    def f1_score(self):
        prec = self.precision()
        recall = self.recall()
        f1 = 2 * prec * recall / (prec + recall)
        return f1

    def dice(self):
        return self.f1_score()

    def kappa(self):
        h = self.mat
        p0 = self.overall_acc()
        pc = np.sum(h.sum(axis=1) * h.sum(axis=0)) / (h.sum())**2
        kappa = (p0 - pc) / (1 - pc)
        return kappa


def eval_seg(pred_fname,
             gt_fname,
             num_classes,
             ignore_index=255,
             metrics='IoU',
             nan_to_num=None):
    """Calculate evaluation metrics for large remote sensing image.

    Args:
        metrics (list[str] | str): Metrics to be evaluated.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.

    Returns:
        float: Overall accuracy on all images.
        ndarray: Per category evalution metrics, shape (num_classes, )
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['IoU', 'Dice', 'F1', 'Prec', 'Recall']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError(f'metrics {metrics} is not supported')

    pred_raster = RasterSampleDataset(pred_fname)
    gt_raster = RasterSampleDataset(gt_fname)
    assert pred_raster.shape == gt_raster.shape, 'Shape mismatch!'

    cmtx = ConfusionMatrix(num_classes, ignore_index)
    for pred, gt in zip(pred_raster, gt_raster):
        pred = pred[0].squeeze()
        gt = gt[0].squeeze()
        cmtx.update(pred, gt)

    overall_acc = cmtx.overall_acc()
    ret_metrics = [overall_acc]
    for metric in metrics:
        if metric == 'IoU':
            iou = cmtx.iou()
            ret_metrics.append(iou)
        elif metric == 'Dice':
            dice = cmtx.dice()
            ret_metrics.append(dice)
        elif metric == 'F1':
            f1 = cmtx.f1_score()
            ret_metrics.append(f1)
        elif metric == 'Prec':
            prec = cmtx.precision()
            ret_metrics.append(prec)
        elif metric == 'Recall':
            recall = cmtx.recall()
            ret_metrics.append(recall)

    if nan_to_num is not None:
        ret_metrics = [
            np.nan_to_num(metric, nan=nan_to_num) for metric in ret_metrics
        ]
    return ret_metrics


if __name__ == '__main__':
    pred_fname = '/data/2021ChengDu/pred/xk1_gray.tif'
    gt_fname = '/data/2021ChengDu/train/label/xk1.tif'
    ret = eval_seg(pred_fname, gt_fname, 5, metrics=['Prec', 'Recall'])
    ret_metrics = eval_seg(pred_fname,
                           gt_fname,
                           num_classes=5,
                           metrics=['IoU', 'Prec', 'Recall'])
    print(ret)
