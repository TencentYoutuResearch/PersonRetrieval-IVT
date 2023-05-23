# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import json


def read_json(fpath):
    with open(fpath, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    return obj


class CUHK_PEDES_LEVEL(object):
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
    def __init__(self, root='data', file='cuhk_pedes_att.json', verbose=True, **kwargs):
        super(CUHK_PEDES_LEVEL, self).__init__()
        self.root = root
        self.file = file

        data_json = read_json(os.path.join(self.root, self.file))

        # reading information
        id_list = sorted(list(set([item['id'] for item in data_json['train']])))            # [11003,]
        id2lb = {item: ii for ii, item in enumerate(id_list)}
        self.train = self._process_dataset(data_json['train'], id2lb)           # [68126,]
        self.val = self._process_dataset(data_json['val'])                      # [6158,]
        self.test = self._process_dataset(data_json['test'])                    # [6156,]
        # self.query = self._process_dataset(data_json['query'])                  # [820,]
        # self.gallery = self._process_dataset(data_json['gallery'])              # [5332,]

        self.num_train_pids, self.num_train_imgs, self.num_train_token = self.get_imagedata_info(self.train)
        self.num_val_pids, self.num_val_imgs, self.num_val_token = self.get_imagedata_info(self.val)
        self.num_test_pids, self.num_test_imgs, self.num_test_token = self.get_imagedata_info(self.test)
        # self.num_query_pids, self.num_query_imgs, self.num_query_token = self.get_imagedata_info(self.query)
        # self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_token = self.get_imagedata_info(self.gallery)

        if verbose:
            print("=> CUHK-PEDES loaded")
            self.print_dataset_statistics()


    def _process_dataset(self, data_list, id2lb=None):
        dataset = []
        for ii, item in enumerate(data_list):
            captions = item['captions']
            file_path = item['file_path']
            file_path = os.path.join(self.root, 'imgs', file_path)
            processed_tokens = item['processed_tokens']
            atts = item['att']
            id = item['id']
            if id2lb is not None:
                id = id2lb[id]
            dataset.append([file_path, id, captions, atts])
        return dataset

    def get_imagedata_info(self, data):
        pids, imgs, tokens = [], [], []
        for img, lab, cap, token in data:
            pids += [lab]
            imgs += [img]
            tokens += [token]
        pids = set(pids)
        imgs = set(imgs)
        num_pids = len(pids)
        num_imgs = len(imgs)
        num_toks = len(tokens)
        return num_pids, num_imgs, num_toks


    def print_dataset_statistics(self,):
        print("Dataset statistics:")
        print("  ----------------------------------------------------")
        print("  subset         | {:>5s}     | {:>5s}     | {:>5s}".format('# ids', '# images', '# tokens'))
        print("  ----------------------------------------------------")
        print("  train          | {:5d}     | {:5d}     | {:5d}".format(self.num_train_pids, self.num_train_imgs, self.num_train_token))
        print("  val            | {:5d}     | {:5d}     | {:5d}".format(self.num_val_pids, self.num_val_imgs, self.num_val_token))
        print("  test           | {:5d}     | {:5d}     | {:5d}".format(self.num_test_pids, self.num_test_imgs, self.num_test_token))
        # print("  query          | {:5d}     | {:5d}     | {:5d}".format(self.num_query_pids, self.num_query_imgs, self.num_query_token))
        # print("  gallery        | {:5d}     | {:5d}     | {:5d}".format(self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_token))







