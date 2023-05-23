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


def read_json(fpath):
    with open(fpath, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    return obj


class RSTPReID(object):
    """
      ----------------------------------------------------
      subset         | # ids     | # images     | # texts
      ----------------------------------------------------
      train          |  3701     | 18505     | 37004
      val            |   200     |  1000     |  2000
      test           |   200     |  1000     |  2000
      query          |   400     |   400     |   400
      gallery        |   400     |  1600     |  1600
    """
    def __init__(self, root='data', file='data_process.json', verbose=True, **kwargs):
        super(RSTPReID, self).__init__()
        self.root = root
        self.file = file

        data_json = read_json(os.path.join(self.root, self.file))

        # reading information
        id_list = sorted(list(set([item['id'] for item in data_json['train']])))            # [3701,]
        id2lb = {item: ii for ii, item in enumerate(id_list)}
        self.train = self._process_dataset(data_json['train'], id2lb)       # [37010,]
        self.val = self._process_dataset(data_json['val'])                  # [2000,]
        self.test = self._process_dataset(data_json['test'])                # [2000,]
        # self.query = self._process_dataset(data_json['query'])              # [400,]
        # self.gallery = self._process_dataset(data_json['gallery'])          # [1600,]

        self.num_train_pids, self.num_train_imgs, self.num_train_token = self.get_imagedata_info(self.train)
        self.num_val_pids, self.num_val_imgs, self.num_val_token = self.get_imagedata_info(self.val)
        self.num_test_pids, self.num_test_imgs, self.num_test_token = self.get_imagedata_info(self.test)
        # self.num_query_pids, self.num_query_imgs, self.num_query_token = self.get_imagedata_info(self.query)
        # self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_token = self.get_imagedata_info(self.gallery)

        if verbose:
            print("=> RSTPReID loaded")
            self.print_dataset_statistics()

    def _split_word(self, datastr):
        datastr = datastr.split(" ")
        result = []
        for itm in datastr:
            if '.' in itm:
                it_s = itm.split('.')
                it = [it for it in it_s if len(it) > 0]
                result += it
                continue
            elif ',' in itm:
                it_s = itm.split(',')
                it = [it for it in it_s if len(it) > 0]
                result += it
                continue
            elif '?' in itm:
                it_s = itm.split('?')
                it = [it for it in it_s if len(it) > 0]
                result += it
                continue
            else:
                result.append(itm)
        return result

    def _process_dataset(self, data_list, id2lb=None, max_tok=90):
        dataset = []
        for ii, item in enumerate(data_list):
            captions = item['captions']
            if len(captions) == 0:
                continue
            file_path = item['img_path']
            file_path = os.path.join(self.root, 'imgs', file_path)
            processed_tokens = self._split_word(captions)
            if len(processed_tokens) > max_tok:
                continue
                # tok = processed_tokens[max_tok]
                # idx = captions.index(tok)
                # captions = captions[0: idx]
                # processed_tokens = processed_tokens[0: max_tok]
            id = item['id']
            if id2lb is not None:
                id = id2lb[id]
            dataset.append([file_path, id, captions, processed_tokens])
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






