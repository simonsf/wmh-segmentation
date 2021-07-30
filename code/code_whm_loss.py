import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function, Variable
from md.mdpytorch.utils.pytorch_version import minor_version

class BinaryDiceLoss(Function):
    """ Dice Loss for binary segmentation
        Dice loss = 1 - Dice (minimize loss, maximize dice)
    """
    def forward(self, input, target, save=True, epsilon=1e-6):

        batchsize = input.size(0)

        # convert probability to binary label using maximum probability
        _, input_label = input.max(1)

        # convert to floats
        input_label = input_label.float()
        target_label = target.float()

        if save:
            # save float version of target for backward
            self.save_for_backward(input, target_label)

        # convert to 1D
        input_label = input_label.view(batchsize, -1)
        target_label = target_label.view(batchsize, -1)

        # compute dice score
        self.intersect = torch.sum(input_label * target_label, 1)
        input_area = torch.sum(input_label, 1)
        target_area = torch.sum(target_label, 1)

        self.sum = input_area + target_area + 2 * epsilon

        # batch dice loss and ignore dice loss where target area = 0
        batch_loss = 1 - 2 * self.intersect / self.sum
        batch_loss[target_area == 0] = 0
        loss = batch_loss.mean()

        if minor_version() < 4:
            return torch.FloatTensor(1).fill_(loss)
        else:
            return loss


class MultiDiceLoss(nn.Module):
    """
    Dice Loss for egmentation(include binary segmentation and multi label segmentation)
    This class is generalization of BinaryDiceLoss
    """
    def __init__(self, weights, num_class):
        """
        :param weights: weight for each class dice loss
        :param num_class: the number of class
        """
        super(MultiDiceLoss, self).__init__()
        self.num_class = num_class

        assert len(weights) == self.num_class, "the length of weight must equal to num_class"
        self.weights = torch.FloatTensor(weights)
        self.weights = self.weights/self.weights.sum()
        self.weights = self.weights.cuda()

    def forward(self, input_tensor, target):
        """
        :param input_tensor: network output tensor
        :param target: ground truth
        :return: weighted dice loss and a list for all class dice loss, expect background
        """
        dice_losses = []
        weight_dice_loss = 0
        all_slice = torch.split(input_tensor, [1] * self.num_class, dim=1)

        for i in range(self.num_class):
            # prepare for calculate label i dice loss
            slice_i = torch.cat([1 - all_slice[i], all_slice[i]], dim=1)
            target_i = (target == i) * 1

            # BinaryDiceLoss save forward information for backward
            # so we can't use one BinaryDiceLoss for all classes
            dice_function = BinaryDiceLoss()
            dice_i_loss = dice_function.forward(slice_i, target_i)

            # save all classes dice loss and calculate weighted dice
            dice_losses.append(dice_i_loss)
            weight_dice_loss += dice_i_loss * self.weights[i]

        return weight_dice_loss, [dice_loss.item() for dice_loss in dice_losses]


class FocalLoss(nn.Module):

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):

        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1) / class_num
        else:
            assert len(alpha) == class_num
            self.alpha = torch.FloatTensor(alpha)
            self.alpha = self.alpha.unsqueeze(1)
            self.alpha = self.alpha / self.alpha.sum()
        self.alpha = self.alpha.cuda()
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.one_hot_codes = torch.eye(self.class_num).cuda()

    def forward(self, input, target):
        # Assume that the input should has one of the following shapes:
        # 1. [sample, class_num]
        # 2. [batch, class_num, dim_y, dim_x]
        # 3. [batch, class_num, dim_z, dim_y, dim_x]
        assert input.dim() == 2 or input.dim() == 4 or input.dim() == 5
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1).contiguous()
            input = input.view(input.numel() // self.class_num, self.class_num)
        elif input.dim() == 5:
            input = input.permute(0, 2, 3, 4, 1).contiguous()
            input = input.view(input.numel() // self.class_num, self.class_num)

        # Assume that the target should has one of the following shapes which
        # correspond to the shapes of the input:
        # 1. [sample, 1] or [sample, ]
        # 2. [batch, 1, dim_y, dim_x] or [batch, dim_y, dim_x]
        # 3. [batch, 1, dim_z, dim_y, dim_x], or [batch, dim_z, dim_y, dim_x]
        target = target.long().view(-1)

        mask = self.one_hot_codes[target.data]
        mask = Variable(mask, requires_grad=False)

        alpha = self.alpha[target.data]
        alpha = Variable(alpha, requires_grad=False)

        probs = (input * mask).sum(1).view(-1, 1) + 1e-10
        log_probs = probs.log()

        if self.gamma > 0:
            batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_probs
        else:
            batch_loss = -alpha * log_probs

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss