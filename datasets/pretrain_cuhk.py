# encoding: utf-8
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


class PRETRAIN_CUHK(object):
    """
    ----------------------------------------------------
    subset         | # ids     | # images     | # tokens
    ----------------------------------------------------
      train          | 11003     | 34054     | 68126
      val            |  1000     |  3078     |  6158
      test           |  1000     |  3074     |  6156
      query          |   820     |   820     |   820
      gallery        |  2000     |  5332     |  5332
    """
    def __init__(self, root='data', file='cuhk_pedes.json', verbose=True, **kwargs):
        super(PRETRAIN_CUHK, self).__init__()
        self.root = root
        self.file = file

        root = r"data_root"
        file_list = [os.path.join(root, item) for item in ['new_cc3m_train.json', 'new_cc3m_val.json', 'new_vg.json', 'new_sbu.json']]

        self.train = []
        for file in file_list:
            # coco: 566747, cc_train: 2530325, cc_val: 8997, vg: 768536, sbu: 857063, total->4731667
            data_json = read_json(file)
            cur_list = self._process_dataset_json(data_json, start=len(self.train))
            self.train += cur_list

        data_json = read_json(os.path.join(self.root, self.file))
        self.test = self._process_dataset(data_json['test'])

        self.num_train_pids, self.num_train_imgs, self.num_train_caps = self.get_imagedata_info(self.train)
        self.num_test_pids, self.num_test_imgs, self.num_test_caps = self.get_imagedata_info(self.test)

        if verbose:
            print("=> PRETRAIN loaded")
            self.print_dataset_statistics()


    def _process_dataset_json(self, data_list, start=0):
        dataset = []
        for ii, item in enumerate(data_list):
            captions = item['caption']
            if len(captions) == 0:
                continue
            file_path = item['image']
            dataset.append([file_path, ii+start, captions])
        return dataset


    def _process_dataset(self, data_list, id2lb=None):
        dataset = []
        for ii, item in enumerate(data_list):
            captions = item['captions']
            file_path = item['file_path']
            file_path = os.path.join(self.root, 'imgs', file_path)
            id = item['id']
            if id2lb is not None:
                id = id2lb[id]
            dataset.append([file_path, id, captions])
        return dataset

    def get_imagedata_info(self, data):
        pids, imgs, caps = [], [], []
        for img, lab, cap in data:
            pids += [lab]
            imgs += [img]
            caps += [cap]
        pids = set(pids)
        imgs = set(imgs)
        num_pids = len(pids)
        num_imgs = len(imgs)
        num_caps = len(caps)
        return num_pids, num_imgs, num_caps


    def print_dataset_statistics(self,):
        print("Dataset statistics:")
        print("  ----------------------------------------------------")
        print("  subset         | {:>5s}     | {:>5s}     | {:>5s}".format('# ids', '# images', '# caps'))
        print("  ----------------------------------------------------")
        print("  train          | {:5d}     | {:5d}     | {:5d}".format(self.num_train_pids, self.num_train_imgs, self.num_train_caps))
        print("  val            | {:5d}     | {:5d}     | {:5d}".format(self.num_test_pids, self.num_test_imgs, self.num_test_caps))







