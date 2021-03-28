import os
import numpy as np
import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod
from BasicalClass import common_ten2numpy, common_predict
from BasicalClass import BasicModule


class BasicUncertainty(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self, instance: BasicModule, device):
        super(BasicUncertainty, self).__init__()
        self.instance = instance
        self.device = device
        self.train_batch_size = self.instance.train_batch_size
        self.test_batch_size = self.instance.test_batch_size
        self.model = self.instance.model.to(self.device)
        self.class_num = self.instance.class_num
        self.save_dir = instance.save_dir

        self.train_y, self.val_y, self.shift1_y, self.shift2_y, self.shift3_y = \
            self.instance.train_y, self.instance.val_y, self.instance.shift1_y, \
                self.instance.shift2_y, self.instance.shift3_y

        self.train_pred_pos, self.train_pred_y =\
            self.instance.train_pred_pos, self.instance.train_pred_y
        self.val_pred_pos, self.val_pred_y = \
            self.instance.val_pred_pos, self.instance.val_pred_y
        self.shift1_pred_pos, self.shift1_pred_y = \
            self.instance.shift1_pred_pos, self.instance.shift1_pred_y
        self.shift2_pred_pos, self.shift2_pred_y = \
            self.instance.shift2_pred_pos, self.instance.shift2_pred_y
        self.shift3_pred_pos, self.shift3_pred_y = \
            self.instance.shift3_pred_pos, self.instance.shift3_pred_y

        self.train_loader = instance.train_loader
        self.val_loader = instance.val_loader
        self.shift1_loader = instance.shift1_loader
        self.shift2_loader = instance.shift2_loader
        self.shift3_loader = instance.shift3_loader

        self.train_num = len(self.train_y)
        self.val_num = len(self.val_y)
        self.shift1_num = len(self.shift1_y)
        self.shift2_num = len(self.shift2_y)
        self.shift3_num = len(self.shift3_y)

        self.train_oracle = np.int32(
            common_ten2numpy(self.train_pred_y).reshape([-1]) == common_ten2numpy(self.train_y).reshape([-1])
        )
        self.val_oracle = np.int32(
            common_ten2numpy(self.val_pred_y).reshape([-1]) == common_ten2numpy(self.val_y).reshape([-1])
        )
        self.shift1_oracle = np.int32(
            common_ten2numpy(self.shift1_pred_y).reshape([-1]) == common_ten2numpy(self.shift1_y).reshape([-1])
        )
        self.shift2_oracle = np.int32(
            common_ten2numpy(self.shift2_pred_y).reshape([-1]) == common_ten2numpy(self.shift2_y).reshape([-1])
        )
        self.shift3_oracle = np.int32(
            common_ten2numpy(self.shift3_pred_y).reshape([-1]) == common_ten2numpy(self.shift3_y).reshape([-1])
        )
        self.softmax = nn.Softmax(dim=1)

    @abstractmethod
    def _uncertainty_calculate(self, data_loader):
        return common_predict(data_loader, self.model, self.device)

    def run(self):
        score = self.get_uncertainty()
        self.save_uncertaity_file(score)
        print('finish score extract for class', self.__class__.__name__)
        return score

    def get_uncertainty(self):
        train_score = self._uncertainty_calculate(self.train_loader)
        val_score = self._uncertainty_calculate(self.val_loader)
        shift1_score = self._uncertainty_calculate(self.shift1_loader)
        shift2_score = self._uncertainty_calculate(self.shift2_loader)
        shift3_score = self._uncertainty_calculate(self.shift3_loader)

        result = {
            'train': train_score,
            'val': val_score,
            'shift1': shift1_score,
            'shift2': shift2_score,
            'shift3': shift3_score,
        }
        return result

    def save_uncertaity_file(self, score_dict):
        data_name = self.instance.__class__.__name__
        uncertainty_type = self.__class__.__name__
        save_name = self.save_dir + '/' + data_name + '/' + uncertainty_type + '.res'
        if not os.path.isdir(os.path.join(self.save_dir, data_name)):
            os.mkdir(os.path.join(self.save_dir, data_name))
        torch.save(score_dict, save_name)
        print('get result for dataset %s, uncertainty type is %s' % (data_name, uncertainty_type))