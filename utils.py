#! /usr/bin/python
# -*- encoding: utf-8 -*-

import numpy
import torch
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import decomposition
from sklearn import manifold
from operator import itemgetter
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):
    
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    tunedThreshold = [];
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);
    
    for tfa in target_fa:
        idx = numpy.nanargmin(numpy.absolute((tfa - fpr))) # numpy.where(fpr<=tfa)[0][-1]
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);
    
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer  = max(fpr[idxE],fnr[idxE])*100
    
    return (tunedThreshold, eer, fpr, fnr);

def plot_images_in_2d(x, y, image_paths, axis=None, zoom=1):
    x, y = numpy.atleast_1d(x, y)
    for x0, y0, image_path in zip(x, y, image_paths):
        image = Image.open("./data/test_shuffle/"+image_path)
        image.thumbnail((100, 100), Image.ANTIALIAS)
        img = OffsetImage(image, zoom=zoom)
        anno_box = AnnotationBbox(img, (x0, y0),
                                  xycoords='data',
                                  frameon=False)
        axis.add_artist(anno_box)
    axis.update_datalim(numpy.column_stack([x, y]))
    axis.autoscale()

def tsne_to_grid_plotter_manual(x, y, selected_filenames):
    S = 2000
    s = 100
    x = (x - min(x)) / (max(x) - min(x))
    y = (y - min(y)) / (max(y) - min(y))
    x_values = []
    y_values = []
    filename_plot = []
    x_y_dict = {}
    for i, image_path in enumerate(selected_filenames):
        a = numpy.ceil(x[i] * (S - s))
        b = numpy.ceil(y[i] * (S - s))
        a = int(a - numpy.mod(a, s))
        b = int(b - numpy.mod(b, s))
        if str(a) + "|" + str(b) in x_y_dict:
            continue
        x_y_dict[str(a) + "|" + str(b)] = 1
        x_values.append(a)
        y_values.append(b)
        filename_plot.append(image_path)
    fig, axis = plt.subplots()
    fig.set_size_inches(22, 22, forward=True)
    plot_images_in_2d(x_values, y_values, filename_plot, zoom=.58, axis=axis)
    plt.show()
    plt.savefig("tSNE.jpg")

def drawTSNE(features):
    pca = decomposition.PCA(50)
    features_keys = list(features.keys())
    features_values = list(features.values())
    features_values = [e[0].tolist() for e in features_values]

    pca.fit(features_values)
    features_compressed = pca.transform(features_values)
    
    num_identities = len(features_keys)
    randomly_selected_index = random.sample(range(num_identities), 300)
    selected_features_keys = [features_keys[idx] for idx in randomly_selected_index]
    selected_features_values = [features_compressed[idx] for idx in randomly_selected_index]

    tsne_results = manifold.TSNE(n_components=2, verbose=1, metric='euclidean').fit_transform(numpy.array(selected_features_values))
    tsne_to_grid_plotter_manual(tsne_results[:,0],tsne_results[:,1], selected_features_keys)

    # tsne_results = manifold.TSNE(n_components=2, verbose=1, metric='euclidean').fit_transform(numpy.array(features_compressed))
    # tsne_to_grid_plotter_manual(tsne_results[:,0],tsne_results[:,1], features_keys)
    return 0


class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        assert len(input.size()) == 2, 'The number of dimensions of input tensor must be 2!'
        # reflect padding to match lengths of in/out
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)