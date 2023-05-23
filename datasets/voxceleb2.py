# encoding: utf-8
"""
@author:  xiujun shu
@contact: shuxj@mail.ioa.ac.cn
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import os.path as osp
import numpy as np
import json
import random
import pickle


class VoxCeleb2(object):
    """
      ----------------------------------------------------
      subset     | train    |    query    |  gallery
      ----------------------------------------------------
      ids        |   727     |    38     |    38
      body       | 28395     |   190     |  5899
      face       | 17143     |   190     |  3114
      audio      |  3757     |   190     |  1255
    """

    def __init__(self, root='data', verbose=True, **kwargs):
        super(VoxCeleb2, self).__init__()
        self.root = root
        self.train_dir = os.path.join(self.root, 'train')
        self.test_dir = os.path.join(self.root, 'test')

        path_pkl = os.path.join(self.root, 'info_voxceleb2.pkl')
        if os.path.exists(path_pkl):
            self.load_info_dict(path_pkl)
        else:
            self.vid_train = [item for item in os.listdir(self.train_dir)]      # 5994
            self.vid_test = [item for item in os.listdir(self.test_dir)]        # 118

            self.p2l_train, self.p2l_test = self.get_pid2label(self.vid_list, self.vid_test)

            # load data
            self.data_train = self._process_data(self.train_dir, self.vid_train, self.p2l_train)  # [28395,]
            self.data_test = self._process_data(self.test_dir, self.vid_test, self.p2l_test)  # [190,], [5899,]

            self.write_info_dict(path_pkl)

        self.num_train_pids = len(list(self.p2l_train.values()))  # 727
        self.num_test_pids = len(list(self.p2l_test.values()))

        if verbose:
            print("=> VoxCeleb2 loaded")
            self.print_dataset_statistics()

    def write_info_dict(self, path_pkl):
        info_dict = {
            'data_train': self.data_train,
            'data_test': self.data_train,
            'p2l_train': self.p2l_train,
            'p2l_test': self.p2l_test,
        }

        fid = open(path_pkl, 'wb')
        pickle.dump(info_dict, fid)
        fid.close()

    def load_info_dict(self, path_pkl):
        with open(path_pkl, 'rb') as fid:
            info_dict = pickle.load(fid)

        self.data_train = info_dict['data_train']
        self.data_test = info_dict['data_test']
        self.p2l_train = info_dict['p2l_train']
        self.p2l_test = info_dict['p2l_test']

        del info_dict


    def get_pid2label(self, vid_train, vid_test):
        pid_container = np.sort(vid_train)
        pid2label_train = {pid: label for label, pid in enumerate(pid_container)}

        pid_container = np.sort(vid_test)
        pid2label_test = {pid: label for label, pid in enumerate(pid_container)}

        return pid2label_train, pid2label_test


    def _process_data(self, root, pid_list, pid2label):
        dataset = []
        camid = 0
        for ii, pid in enumerate(pid_list):
            path_p = os.path.join(root, pid)
            view = os.listdir(path_p)
            label = pid2label[view]
            for vw in view:
                path_i = os.path.join(path_p, vw)
                files = os.listdir(path_i)
                for file in files:
                    dataset.append([os.path.join(pid, vw, file), label, camid, 1])
                    camid += 1
        return dataset


    def print_dataset_statistics(self, ):
        print("Dataset statistics:")
        print("  ----------------------------------------------------")
        print("  subset     | {:>5s}    | {:>8s}".format('train', 'test'))
        print("  ----------------------------------------------------")
        print("  ids        | {:5d}     | {:5d}".format(self.num_train_pids, self.num_test_pids))
















