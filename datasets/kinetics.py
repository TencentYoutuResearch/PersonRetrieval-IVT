# encoding: utf-8
"""
@author:  xiujun shu
@contact: shuxj@mail.ioa.cn
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import json
import pickle

def read_json(fpath):
    with open(fpath, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    return obj


class KINETICS400(object):
    """
      ----------------------------------------------------
      subset         | # ids     | # images     | # caps
      ----------------------------------------------------
      train    |  5000     | 71298     | 143438
      val     |  5836     |  5836     | 11672
    """
    def __init__(self, root='data', verbose=True, **kwargs):
        super(KINETICS400, self).__init__()
        self.path_train = os.path.join(root[0], 'train_256_flow')
        self.path_val = os.path.join(root[1], 'val_rawframes_flow')

        info_train = read_json(os.path.join(root[0], 'train_256_flow.json'))
        info_val = read_json(os.path.join(root[1], 'val_rawframes_flow.json'))

        classes = sorted(list(info_train.keys())) 
        cls2lab = {cls: ii for ii, cls in enumerate(classes)}
        self.train = self._process_dataset_train(info_train, cls2lab)   # [234584,]
        self.val = self._process_dataset_val(info_val, cls2lab)         # [15181,]

        self.num_train_cls, self.num_train_imgs = self.get_imagedata_info(self.train)
        self.num_val_cls, self.num_val_imgs = self.get_imagedata_info(self.val)

        if verbose:
            print("=> KINETICS400 loaded")
            self.print_dataset_statistics()

    def sort_by_str(self, str_list):
        int_list = [int(item) for item in str_list]
        index = np.argsort(int_list)
        result = [str_list[idx] for idx in index]
        return result



    def _process_dataset_json(self, data_list, start=0):
        dataset = []
        for ii, item in enumerate(data_list):
            captions = item['caption']
            if len(captions) == 0:
                continue
            file_path = item['image']
            dataset.append([file_path, ii+start, captions])
        return dataset

    def _process_dataset_train(self, data_dict, cls2lab=None):
        result = []
        for ii, cls in enumerate(data_dict):
            label = cls2lab[cls]
            for vid in data_dict[cls]:
                images = data_dict[cls][vid]
                result.append([self.path_train, label, images])
        return result

    def _process_dataset_val(self, data_dict, cls2lab=None):
        result = []
        for ii, cls in enumerate(data_dict):
            label = cls2lab[cls]
            for vid in data_dict[cls]:
                images = data_dict[cls][vid]
                result.append([self.path_val, label, images])
        return result

    def get_imagedata_info(self, data):
        cls, imgs = [], []
        for _, lab, img in data:
            cls += [lab]
        cls = set(cls)
        num_cls = len(cls)
        num_img = len(data)
        return num_cls, num_img


    def print_dataset_statistics(self,):
        print("Dataset statistics:")
        print("  ----------------------------------------------------")
        print("  subset   | {:>5s}    | {:>5s}".format('# ids', '# images'))
        print("  ----------------------------------------------------")
        print("  train    | {:5d}     | {:5d}".format(self.num_train_cls, self.num_train_imgs))
        print("  val      | {:5d}     | {:5d}".format(self.num_val_cls, self.num_val_imgs))








