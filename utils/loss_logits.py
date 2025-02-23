import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = filterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss.
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                    for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss for flattened inputs.
    """
    if probas.numel() == 0:
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes == 'present' and fg.sum() == 0):
            continue
        class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch.
    """
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore=0):
        """
        Lovasz-Softmax loss initialization.

        Args:
            classes (str or list): 'all' for all classes, 'present' for present classes, or list of specific classes.
            per_image (bool): Calculate per image or per batch.
            ignore (int): Class label to ignore (unlabeled class).
        """
        super(LovaszSoftmax, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, logits, labels):
        """
        Forward pass of Lovasz-Softmax loss.
        Args:
            logits: [B, C, H, W] Logits of the network (before softmax).
            labels: [B, H, W] Ground truth labels.
        """
        probas = torch.softmax(logits, dim=1)
        return lovasz_softmax(probas, labels, self.classes, self.per_image, self.ignore)

waymovalidfreqs = [160951307,
 69475897,
 7765250,
 3272170,
 1828020,
 503,
 248386,
 5913634,
 4865385,
 502838,
 8645906,
 488421,
 171792,
 197329,
 243989204,
 159686476,
 13575661,
 11267605,
 190620777,
 5444214,
 4198428,
 74503790,
 45916607]

#Balance
waymovalidfreqs = [f+20000 for f in waymovalidfreqs]


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_frequencies=waymovalidfreqs, ignore=0, device='cuda'):
        """
        Args:
            class_frequencies (list or tensor): Frequency of each class.
            ignore (int): Class label to ignore.
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore = ignore
        
        # Calculate weights as 1 / sqrt(freq)
        class_frequencies = torch.tensor(class_frequencies, dtype=torch.float).to(device)
        weights = 1.0 / torch.sqrt(class_frequencies)
        weights[ignore] = weights[ignore]/10  # Set ignore class weight to 0
        weights = weights/sum(weights)
        self.weights = weights

    def forward(self, logits, labels):
        """
        Forward pass of weighted cross-entropy loss.
        Args:
            logits: [B, C, H, W] Logits from the network (before softmax).
            labels: [B, H, W] Ground truth labels.
        """
        return F.cross_entropy(logits, labels, weight=self.weights, ignore_index=self.ignore)


class Lovasz_WCE(nn.Module):
    def __init__(self, class_frequencies=waymovalidfreqs, lovasz_weight=1, cross_entropy_weight=0.4, ignore=0):
        """
        Args:
            class_frequencies (list or tensor): Frequency of each class for weighted cross-entropy.
            lovasz_weight (float): Weight of Lovasz loss in the combined loss.
            cross_entropy_weight (float): Weight of cross-entropy loss in the combined loss.
            ignore (int): Class label to ignore (unlabeled class).
        """
        super(Lovasz_WCE, self).__init__()
        self.lovasz_loss = LovaszSoftmax(ignore=ignore)  # From previous code
        self.cross_entropy_loss = WeightedCrossEntropyLoss(class_frequencies, ignore=ignore)
        self.lovasz_weight = lovasz_weight
        self.cross_entropy_weight = cross_entropy_weight

    def forward(self, logits, labels):
        """
        Compute combined loss.
        Args:
            logits: [B, C, H, W] Logits from the network.
            labels: [B, H, W] Ground truth labels.
        """
        # Calculate both losses
        ce_loss = self.cross_entropy_loss(logits, labels)
        lovasz_loss = self.lovasz_loss(logits, labels)
        
        # Weighted sum of both losses
        combined_loss = self.lovasz_weight * lovasz_loss + self.cross_entropy_weight * ce_loss
        return combined_loss

