
import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.metrics import accuracy_score


def readuea(dir, fname, mode):
    X = np.load(os.path.join(dir, fname, "{}_{}.npy".format(fname, mode)))
    Y = np.load(os.path.join(dir, fname, "{}_{}_label.npy".format(fname, mode)))
    return X, Y


def train_test_supervised_batch(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        dataloader,
        device,
        optimizer=None,
        is_train=True,
        comparison_key='tcrm2_out'
):
    prev_grad_status = torch.is_grad_enabled()
    running_loss = 0.
    preds, gts = [], []
    if is_train:
        model.train()
    else:
        model.eval()
        torch.set_grad_enabled(False)
    try:
        for i, (xs, labels, _, _) in enumerate(dataloader, 0):
            xs, labels = xs.to(device), labels.to(device)
            out = model(xs)
            if comparison_key in out:
                out = out[comparison_key]
            if criterion.reduction == 'none':
                loss = sum(criterion(out, labels)) / out.shape[0]
            else:
                loss = criterion(out, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)
                optimizer.step()
            running_loss += loss.item()
            preds.append(out.argmax(dim=-1).detach().to('cpu').numpy())
            gts.append(labels.detach().to('cpu').numpy())
    except Exception as e:
        print(e)
        torch.set_grad_enabled(prev_grad_status)
    torch.set_grad_enabled(prev_grad_status)

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)

    acc = accuracy_score(y_true=gts, y_pred=preds)
    running_loss = running_loss / (i + 1)

    return {
        'acc': acc,
        'loss': running_loss,
    }


def train_semi_supervised_batch(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        dl_labeled,
        device,
        train_config,
        unlabeled_dataloader,
        optimizer=None,

):
    model.train()
    tcrm2_running_loss, ttrm2_running_loss = 0., 0.
    train_pred_ttrm2, train_pred_tcrm2, gts = [], [], []
    for i, (l_xs, l_labels, l_is_labeled, _) in enumerate(dl_labeled, 0):
        unlabeled_data = unlabeled_dataloader.__next__()
        unlabeled_xs = unlabeled_data['xs']
        xs = torch.cat([l_xs, unlabeled_xs], dim=0)

        xs, l_labels = xs.to(device), l_labels.to(device)

        labeled_xs_num = l_xs.shape[0]
        out_dict = model(xs, used_ttrm2= True, alpha=train_config.alpha, beta=train_config.beta)

        # modeling the relationship between time series and classes
        out_tcrm2 = out_dict['tcrm2_out']
        softmax_out_tcrm2 = torch.softmax(out_tcrm2, dim=-1)
        pred_prob_tcrm2, pred_tcrm2 = softmax_out_tcrm2.max(dim=-1)

        # modeling the relationship between time series and time series
        out_ttrm2 = out_dict['ttrm2_out']
        softmax_out_ttrm2 = torch.softmax(out_ttrm2, dim=-1)
        pred_prob_ttrm2, pred_ttrm2 = softmax_out_ttrm2.max(dim=-1)

        if labeled_xs_num>0:
            losses_tcrm2 = criterion(out_tcrm2[:labeled_xs_num], l_labels)
            losses_ttrm2 = criterion(out_ttrm2[:labeled_xs_num], l_labels)
            loss = torch.mean(losses_tcrm2)+ train_config.lambda_loss*torch.mean(losses_ttrm2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)
            optimizer.step()

            tcrm2_running_loss += torch.mean(losses_tcrm2).item()
            ttrm2_running_loss += torch.mean(losses_ttrm2).item()

            gts.append(l_labels.detach().to('cpu').numpy())
            train_pred_ttrm2.append(pred_ttrm2[:labeled_xs_num].detach().to('cpu').numpy())
            train_pred_tcrm2.append(pred_tcrm2[:labeled_xs_num].detach().to('cpu').numpy())

    train_pred_ttrm2 = np.concatenate(train_pred_ttrm2, axis=0)
    train_pred_tcrm2 = np.concatenate(train_pred_tcrm2, axis=0)
    gts = np.concatenate(gts, axis=0)

    acc_tcrm2 = accuracy_score(y_true=gts, y_pred=train_pred_tcrm2)
    acc_ttrm2 = accuracy_score(y_true=gts, y_pred=train_pred_ttrm2)
    tcrm2_running_loss = tcrm2_running_loss / (i + 1)
    ttrm2_running_loss = ttrm2_running_loss / (i + 1)

    return {
        'acc_tcrm2': acc_tcrm2,
        'acc_ttrm2': acc_ttrm2,
        'tcrm2_loss': tcrm2_running_loss,
        'ttrm2_loss': ttrm2_running_loss,
    }


def train_supervised_batch_with_cl(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        criterion_cl: torch.nn.Module,
        dataloader, device,
        train_config,
        optimizer=None,
):
    model.train()
    origin_running_loss, cl_running_loss = 0., 0.
    train_pred, gts = [], []
    for i, (xs, labels, _, _) in enumerate(dataloader, 0):
        xs, labels = xs.to(device), labels.to(device)
        out_dict = model(xs)

        # modeling the relationship between time series and classes
        out, features = out_dict['origin_out'], out_dict['x_embeddings']
        softmax_out = torch.softmax(out, dim=-1)
        pred_prob, pred = softmax_out.max(dim=-1)

        if criterion.reduction == 'none':
            loss_ce = sum(criterion(out, labels)) / out.shape[0]
        else:
            loss_ce = criterion(out, labels)
        losses_cl = criterion_cl(features, labels)

        loss = loss_ce + train_config.lambda_loss * losses_cl['center_loss']
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)
        optimizer.step()

        origin_running_loss += loss_ce.item()
        cl_running_loss += losses_cl['center_loss'].item()

        gts.append(labels.detach().to('cpu').numpy())
        train_pred.append(pred.detach().to('cpu').numpy())

    train_pred = np.concatenate(train_pred, axis=0)
    gts = np.concatenate(gts, axis=0)

    acc = accuracy_score(y_true=gts, y_pred=train_pred)
    origin_running_loss = origin_running_loss / (i + 1)
    cl_running_loss = cl_running_loss / (i + 1)

    return {
        'acc': acc,
        'origin_running_loss': origin_running_loss,
        'cl_running_loss': cl_running_loss,
    }



"""
The code about Center Loss is from https://github.com/KaiyangZhou/pytorch-center-loss.
"""
class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, device = 'cpu'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        self.to(self.device)
    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        classes = torch.arange(self.num_classes).long().to(self.device)

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return {
            "center_loss": loss
        }
