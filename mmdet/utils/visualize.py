import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import os

from PIL import Image
from torchvision.transforms import ToTensor
from matplotlib.patches import Rectangle


dirname = ''


def imsave(image, title=None, save=None):
    plt.imshow(image, cmap='gray')
    if title:
        plt.title(title)
    if save:
        plt.savefig(f"{dirname}/{save}.png")


def bincount(data, num_bins):
    min_, max_ = data.min(), data.max()
    # x = torch.linspace(min_, max_, steps=num_bins)
    # return data.bincount(x)
    # return torch.bincount(data, minlength=2)
    return torch.histc(data, bins=num_bins, min=float(min_), max=float(max_))


def multi_imsave(image, rows, cols, save=None):
    plt.figure(figsize=(14, 10))
    i = 0
    save_, title_ = None, None
    for row in range(rows):
        for col in range(cols):
            if (row == rows-1) and (col == cols-1):
                save_ = save
            plt.subplot(rows, cols, i+1)
            count = bincount(image[i].reshape(-1), 2)
            torch.set_printoptions(precision=3, sci_mode=False)
            imsave(image[i].cpu().detach().numpy(), f"{count}", save_)
            torch.set_printoptions(precision=None, sci_mode=True)
            i = i + 1


COLOR_CODE = ['#FF5A5A', '#DC9146', '#FFCD28', '#FAFAA0', '#CBFF75', '#AFFFEE', '#87F5F5', '#5AD2FF', '#A390EE']
EDGE_COLOR_CODE = ['#CD0000', '#8B4513', '#FF8200', '#FFC81E', '#64CD3C', '#66CDAA', '#20B2AA', '#0000FF',
                   '#6A5ACD']
EPS = 1e-2


def get_file_name(debug_cfg, name, extension='png', img_meta=None):
    out_dir = debug_cfg['out_dir']
    # corruption = debug_cfg['corruption']
    # severity = debug_cfg['corruption_severity']
    if img_meta:
        name = f"{img_meta['ori_filename'].split('.png')[0]}_{name}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    _ = name.split("/")[0]
    if not os.path.exists(f"{out_dir}/{_}"):
        os.makedirs(f"{out_dir}/{_}")
    return f"{out_dir}/{name}.{extension}"


def visualize_score_distribution(scores, name, img_meta=None, debug_cfg=None, bins=100, alpha=1):
    '''
    Args:
        scores: Tensor(N, 1)
            e.g.1, Given proposal_list={list:1} ∋ (1000,5),
                    inputs proposal_list[0][:,4]
            e.g.2, Given
        name: str
            e.g., 'proposal_list'
        img_meta: dict
        debug_cfg: dict
    Returns:

    '''
    if not torch.is_tensor(scores):
        scores = torch.Tensor(scores)
    if scores.dim() == 1:
        scores = scores.reshape(scores.shape[0], 1)
    if scores.shape[-1] != 1:
        raise ValueError("The input scores should be shaped as (N, 1),"
                         f"but got {scores.shape}.")
    try:
        np_scores = np.array(scores.squeeze().to('cpu').detach())
        counts, edges, bars = plt.hist(np_scores, bins=bins, alpha=alpha)
        plt.bar_label(bars)
        plt.xlim(0.0, 1.0)
        plt.title(f"range=({np_scores.min():.2f},{np_scores.max():.2f})")
        if debug_cfg and (name in debug_cfg['save_list']):
            fn = get_file_name(debug_cfg, name, img_meta=img_meta)
            plt.savefig(fn)
    except: # when np_scores is empty
        pass
    plt.close()


def visualize_score_distribution_stacked(scores, name, img_meta=None, debug_cfg=None, bins=100, alpha=1):
    '''
    Args:
        scores_list: [Tensor(N, 1), Tensor(N, 1), ...]
            e.g.1, Given proposal_list={list:1} ∋ (1000,5),
                    inputs proposal_list[0][:,4]
            e.g.2, Given
        name: str
            e.g., 'proposal_list'
        img_meta: dict
        debug_cfg: dict
    Returns:

    '''
    if not torch.is_tensor(scores):
        scores = torch.Tensor(scores)
    if scores.dim() == 1:
        scores = scores.reshape(scores.shape[0], 1)
    if scores.shape[-1] != 1:
        raise ValueError("The input scores should be shaped as (N, 1),"
                         f"but got {scores.shape}.")
    try:
        np_scores = np.array(scores.squeeze().to('cpu').detach())
        counts, edges, bars = plt.hist(np_scores, bins=bins, alpha=alpha)
        plt.bar_label(bars)
        plt.xlim(0.0, 1.0)
        plt.title(f"range=({np_scores.min():.2f},{np_scores.max():.2f})")
        if debug_cfg and (name in debug_cfg['save_list']):
            fn = get_file_name(debug_cfg, name, img_meta=img_meta)
            plt.savefig(fn)
    except: # when np_scores is empty
        pass
    plt.close()


def visualize_score_density(fives, name, topk=None, save_original=False, img_meta=None, debug_cfg=None, win_name=''):
    '''
    Args:
        fives: (N,5) or {list} ∋ (N,5)
                where 5 contains [tl_x, tl_y, br_x, br_y, score]
        name: (str) e.g., 'proposal_list_score_density'
        save_original: (bool) save the original image if True (default=False).
        img_meta: (dict)
        debug_cfg: (dict)
        win_name: (str)
        alpha:

    Returns:

    '''
    with Image.open(img_meta['filename']) as img:
        fig = plt.figure(win_name)
        dpi = fig.get_dpi()
        width, height = img_meta['ori_shape'][1], img_meta['ori_shape'][0]
        fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = plt.gca()
        ax.axis('off')

        tf_toTensor = ToTensor()
        img_tensor = tf_toTensor(img).permute(1, 2, 0)  # (3, 1024, 2048) -> (1024, 2048, 3)

        plt.imshow(img_tensor)
        if save_original and debug_cfg:
            fn = get_file_name(debug_cfg, f"{name}_original", img_meta=img_meta)
            plt.savefig(fn)

        if torch.is_tensor(fives):
            if topk:
                _, indices = torch.sort(fives[:, 4], -1, descending=True)
                fives = fives[indices]
            for i in range(fives.shape[0]):
                if i > topk:
                    break
                tl_x, tl_y, br_x, br_y, score = fives[i, :]
                x, y = int(tl_x.item()), int(tl_y.item())
                w, h = int((br_x - tl_x).item()), int((br_y - tl_y).item())
                patch = Rectangle((x, y), w, h, facecolor='red', alpha=score.item())
                ax.add_patch(patch)
        elif isinstance(fives, list):
            num_classes = len(fives)
            for c in range(num_classes):
                bbox_result = fives[c]
                for i in range(bbox_result.shape[0]):
                    tl_x, tl_y, br_x, br_y, score = bbox_result[i, :]
                    x, y = int(tl_x.item()), int(tl_y.item())
                    w, h = int((br_x - tl_x).item()), int((br_y - tl_y).item())
                    patch = Rectangle((x, y), w, h, facecolor=COLOR_CODE[c], alpha=score.item())
                    ax.add_patch(patch)
            if 'annotations' in debug_cfg:
                labels = debug_cfg['annotations']['labels']
                bboxes = debug_cfg['annotations']['bboxes']
                for i in range(labels.shape[0]):
                    tl_x, tl_y, br_x, br_y = bboxes[i, :]
                    x, y = int(tl_x.item()), int(tl_y.item())
                    w, h = int((br_x - tl_x).item()), int((br_y - tl_y).item())
                    patch = Rectangle((x, y), w, h, edgecolor=EDGE_COLOR_CODE[labels[i]], facecolor='none')
                    ax.add_patch(patch)
        else:
            raise TypeError(f'fives must be Tensor or list,'
                            f'but got {fives.dtype}.')

        if debug_cfg and (name in debug_cfg['save_list']):
            fn = get_file_name(debug_cfg, name, img_meta=img_meta)
            plt.savefig(fn)
        plt.close()


def visualize_image(img_meta, name, debug_cfg=None, win_name=''):
    '''
    Args:
        img_meta: (dict)
        name: (str) e.g., 'proposal_list_score_density'
        save_original: (bool) save the original image if True (default=False).
        debug_cfg: (dict)
        win_name: (str)
    '''
    with Image.open(img_meta['filename']) as img:
        fig = plt.figure(win_name)
        dpi = fig.get_dpi()
        width, height = img_meta['ori_shape'][1], img_meta['ori_shape'][0]
        fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = plt.gca()
        ax.axis('off')

        tf_toTensor = ToTensor()
        img_tensor = tf_toTensor(img).permute(1, 2, 0)  # (3, 1024, 2048) -> (1024, 2048, 3)

        plt.imshow(img_tensor)
        if name in debug_cfg['save_list']:
            fn = get_file_name(debug_cfg, f"{name}", img_meta=img_meta)
            plt.savefig(fn)
        plt.close()


from sklearn.manifold import TSNE
from matplotlib import cm
from sklearn import decomposition


def plot_tsne(test_features, targets=None, title=None, save=None):
    test_features = test_features.cpu().detach().numpy().data
    y = targets.cpu().detach().numpy().data

    tsne = TSNE(n_components=2, perplexity=10, n_iter=300, learning_rate=200.0, init='random')
    tsne_ref = tsne.fit_transform(test_features)

    fig = plt.figure(figsize=(12, 12))
    plt.scatter(tsne_ref[:, 0], tsne_ref[:, 1], marker='.',
                cmap=cm.Paired, c=y)
    if title is not None:
        plt.title(f't-SNE ({title})', weight='bold').set_fontsize('14')
    plt.xlabel('x', weight='bold').set_fontsize('10')
    plt.ylabel('y', weight='bold').set_fontsize('10')
    plt.axis('equal')
    if save is not None:
        plt.savefig(save)

    return plt, fig

def single_plot_tsne(test_features, targets=None, title=None,
                     n_components=2, perplexity=10, n_iter=300, learning_rate=200.0):
    test_features = test_features.cpu().detach().numpy().data
    y = targets.cpu().detach().numpy().data

    pca = decomposition.PCA(n_components=9)
    reduced_test_features = pca.fit_transform(test_features)

    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, learning_rate=learning_rate, init='random')
    tsne_ref = tsne.fit_transform(reduced_test_features)

    plt.scatter(tsne_ref[:, 0], tsne_ref[:, 1], marker='.', cmap='tab10', c=y)
    if title is not None:
        plt.title(f"t-SNE for {title}: ({n_components}, {perplexity}, {n_iter}, {learning_rate}, random)", weight='bold').set_fontsize('12')
    plt.xlabel('x', weight='bold').set_fontsize('10')
    plt.ylabel('y', weight='bold').set_fontsize('10')
    cbar = plt.colorbar()
    cbar.set_ticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
    bincounts = torch.bincount(targets.type(torch.int)).tolist()
    cityscapes_classes = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'background']
    ticklabel_list = []
    for i in range(len(bincounts)):
        ticklabel_list.append(f"{cityscapes_classes[i]}({bincounts[i]})")
    cbar.set_ticklabels(ticklabel_list)
    plt.axis('equal')

    return plt

# from sklearn.manifold import TSNE
# from matplotlib import cm
# from sklearn import decomposition
# test_features = all_bbox_feats_.cpu().detach().numpy().data
# y = all_labels_.cpu().detach().numpy().data
# title = 'bbox_feats'
#
# pca = decomposition.PCA(n_components=9)
# reduced_test_features = pca.fit_transform(test_features)
#
#
# plt.clf()
# tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, learning_rate=learning_rate, init='random')
# tsne_ref = tsne.fit_transform(reduced_test_features)
# plt.scatter(tsne_ref[:, 0], tsne_ref[:, 1], marker='.', cmap='tab10', c=y)
# plt.title(f"t-SNE: (n_components={n_components}, perplexity={perplexity}, n_iter={n_iter}, lr={learning_rate}, random)", weight='bold').set_fontsize('8')
# # plt.savefig(f'/ws/data/oadg/debug/given/outs/gaussian_noise/0/frankfurt/t-SNE({n_components},{perplexity},{n_iter},{learning_rate}).png')

import os


def multi_plot_tsne(test_features_list, targets_list=None, title_list=None, rows=1, cols=1, save=None,
                    n_components=2, perplexity=10, n_iter=300):
    fig = plt.figure(figsize=(8*cols, 7*rows))
    i = 0
    for row in range(rows):
        for col in range(cols):
            plt.subplot(rows, cols, i+1)
            single_plot_tsne(test_features_list[i], targets_list[i], title_list[i],
                             n_components=n_components, perplexity=perplexity, n_iter=n_iter)
            i = i + 1
    if save is not None:
        filename = save.split('/')[-1]
        dirname = save.replace(filename, '')
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        plt.savefig(save)

    return plt, fig

# import matplotlib.pyplot as plt
# cols, rows = 1, 1
# fig = plt.figure(figsize=(8*cols, 7*rows))
#
# filename = save.split('/')[-1]
# dirname = save.replace(filename, '')
# if not os.path.exists(dirname):
# os.makedirs(dirname)
# plt.savefig(save)


def plot_matrix(cm,
                dataset='cityscapes',
                classes=0,
                normalize='None',
                txt=True,
                title='Matrix',
                cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if classes != 0:
        classes = [i for i in range(classes)]
    elif dataset == 'cityscapes':
        classes = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'background']
    elif dataset == 'coco':
        classes = []

    plt.figure(figsize=(len(classes), len(classes)))
    if normalize == 'None':
        pass
    elif normalize == 'x':
        cm = cm.astype('float') / (cm.sum(axis=0)[:, np.newaxis] + 1e-8)
        print("x Normalized confusion matrix")
    elif normalize == 'y':
        cm = cm.astype('float') / (cm.sum(axis=1)[np.newaxis, :] + 1e-8)
        print("y Normalized confusion matrix")
    elif normalize == 'xy':
        cm = cm.astype('float') / (cm.sum())
        print("XY Normalized confusion matrix")

    else:
        # print('Confusion matrix, without normalization')
        pass

    # print(cm)
    # print(cm.diag() / cm.sum(1))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if txt:
        fmt = '.2f' if normalize else '.2f' #'d'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('anchor class')
    plt.xlabel('Contrast class')
    # wandb.log({title: plt})
    return plt
    # plt.savefig('/ws/external/visualization_results/confusion_matrix.png')


def plot_bar(feature,
             normalize='None',
             txt=True,
             title='1D plot feature'
             ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    x = [i for i in range(np.shape(feature)[0])]

    plt.figure()

    # if normalize == 'None':
    #     pass
    # elif normalize == 'y':
    #     cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    #     print("Y Normalized confusion matrix")
    # elif normalize == 'xy':
    #     cm = cm.astype('float') / (cm.sum())
    #     print("XY Normalized confusion matrix")
    #
    # else:
    #     # print('Confusion matrix, without normalization')
    #     pass

    # print(cm)
    # print(cm.diag() / cm.sum(1))

    plt.bar(x, height=feature)
    plt.title(title)

    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)

    # if txt:
    #     fmt = '.2f' if normalize else '.2f' #'d'
    #     thresh = cm.max() / 2.
    #     for i in range(cm.shape[0]):
    #         for j in range(cm.shape[1]):
    #             plt.text(j, i, format(cm[i, j], fmt),
    #                      horizontalalignment="center",
    #                      color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('value')
    plt.xlabel('feature dim')

    return plt
    # plt.savefig('/ws/external/visualization_results/confusion_matrix.png')