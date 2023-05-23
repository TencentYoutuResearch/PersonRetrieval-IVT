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


class VoxCeleb2_C(object):
    """
     subset     | train    |     test
      -------------------------------
      ids        |  6112     |  6112
    """

    def __init__(self, root='data', verbose=True, **kwargs):
        super(VoxCeleb2_C, self).__init__()
        self.root = root
        self.train_dir = os.path.join(self.root, 'train')
        self.test_dir = os.path.join(self.root, 'test')

        path_pkl = os.path.join(self.root, 'info_voxceleb2.pkl')
        if os.path.exists(path_pkl):
            self.load_info_dict(path_pkl)
        else:
            self.vid_train = [item for item in os.listdir(self.train_dir)]      # 5994
            self.vid_test = [item for item in os.listdir(self.test_dir)]        # 118
            self.vid_list = list(set(self.vid_train + self.vid_test))           # 6112

            self.pid2label = self.get_pid2label(self.vid_list)

            # load data
            data_train1, data_test1 = self._process_data(self.train_dir, 'train', self.vid_train, self.pid2label)  # [28395,]
            data_train2, data_test2 = self._process_data(self.test_dir, 'test', self.vid_test, self.pid2label, len(data_train1 + data_test1))     # [190,], [5899,]
            self.data_train = data_train1 + data_train2         # [1097686,]
            self.data_test = data_test1 + data_test2            # [30560,]

            self.write_info_dict(path_pkl)

        self.num_pids_train = len(set([item[1] for item in self.data_train]))   # 6112
        self.num_pids_test = len(set([item[1] for item in self.data_test]))     # 6112

        if verbose:
            print("=> VoxCeleb2 loaded")
            self.print_dataset_statistics()

    def write_info_dict(self, path_pkl):
        info_dict = {
            'data_train': self.data_train,
            'data_test': self.data_test,
            'pid2label': self.pid2label,
            'vid_list': self.vid_list,
        }

        fid = open(path_pkl, 'wb')
        pickle.dump(info_dict, fid)
        fid.close()

    def load_info_dict(self, path_pkl):
        with open(path_pkl, 'rb') as fid:
            info_dict = pickle.load(fid)

        self.data_train = info_dict['data_train']
        self.data_test = info_dict['data_test']
        self.pid2label = info_dict['pid2label']
        self.vid_list = info_dict['vid_list']

        del info_dict


    def get_pid2label(self, vid_list):
        pid_container = np.sort(vid_list)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        return pid2label


    def _process_data(self, root, data_mode, pid_list, pid2label, restart=0):
        data_train = []
        data_test = []
        camid = restart
        for ii, pid in enumerate(pid_list):
            path_p = os.path.join(root, pid)
            view = os.listdir(path_p)
            label = pid2label[pid]
            data_ii = []
            for vw in view:
                path_i = os.path.join(path_p, vw)
                files = os.listdir(path_i)
                for file in files:
                    data_ii.append([os.path.join(data_mode, pid, vw, file), label, camid, 1])
                    camid += 1
            random.shuffle(data_ii)
            data_train += data_ii[:-5]
            data_test += data_ii[-5:]
        return data_train, data_test


    def print_dataset_statistics(self, ):
        print("Dataset statistics:")
        print("  ----------------------------------------------------")
        print("  subset     | {:>5s}    | {:>8s}".format('train', 'test'))
        print("  ----------------------------------------------------")
        print("  ids        | {:5d}     | {:5d}".format(self.num_pids_train, self.num_pids_test))
















