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


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []
        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def get_videodata_info(self, data, return_tracklet_info=False):
        pids, cams, tracklet_info = [], [], []
        for img_paths, pid, camid in data:
            pids += [pid]
            cams += [camid]
            tracklet_info += [len(img_paths)]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_tracklets = len(data)
        if return_tracklet_info:
            return num_pids, num_tracklets, num_cams, tracklet_info
        return num_pids, num_tracklets, num_cams

    def print_dataset_statistics(self):
        raise NotImplementedError



class LaST(BaseDataset):
    """
      --------------------------------------
      subset         | # ids     | # images
      --------------------------------------
      train          |  5000     |    71248
      query          |    56     |      100
      gallery        |    57     |    21279
      query_test     |  5805     |    10176
      gallery_test   |  5807     |   125353
    """
    dataset_dir = ""

    def __init__(self, root='data', verbose=True, **kwargs):
        super(LaST, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'val', 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'val', 'gallery')
        self.query_test_dir = osp.join(self.dataset_dir, 'test', 'query')
        self.gallery_test_dir = osp.join(self.dataset_dir, 'test', 'gallery')

        self._check_before_run()

        self.pid2label = self.get_pid2label(self.train_dir)
        self.train = self._process_dir(self.train_dir, pid2label=self.pid2label, relabel=True)
        self.query = self._process_dir(self.query_dir, relabel=False)
        self.gallery = self._process_dir(self.gallery_dir, relabel=False, recam=len(self.query))
        self.query_test = self._process_dir(self.query_test_dir, relabel=False)
        self.gallery_test = self._process_dir(self.gallery_test_dir, relabel=False, recam=len(self.query_test))

        if verbose:
            print("=> LaST loaded")
            self.print_dataset_statistics_movie(self.train, self.query, self.gallery, self.query_test, self.gallery_test)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
        self.num_query_test_pids, self.num_query_test_imgs, self.num_query_test_cams, self.num_query_test_vids = self.get_imagedata_info(self.query_test)
        self.num_gallery_test_pids, self.num_gallery_test_imgs, self.num_gallery_test_cams, self.num_gallery_test_vids = self.get_imagedata_info(self.gallery_test)


    def get_pid2label(self, dir_path):
        persons = os.listdir(dir_path)
        pid_container = list(set([int(pid) for pid in persons]))
        pid_container = np.sort(pid_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        return pid2label

    # def get_pid2label(self, dir_path):
    #     img_paths = glob.glob(osp.join(dir_path, '*/*.jpg'))            # [103367,]
    #
    #     pid_container = set()
    #     for img_path in img_paths:
    #         pid = int(os.path.basename(img_path).split('_')[0])
    #         pid_container.add(pid)
    #     pid_container = np.sort(list(pid_container))
    #     pid2label = {pid: label for label, pid in enumerate(pid_container)}
    #     return pid2label

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        if not osp.exists(self.query_test_dir):
            raise RuntimeError("'{}' is not available".format(self.query_test_dir))
        if not osp.exists(self.gallery_test_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_test_dir))

    def _process_dir(self, dir_path, pid2label=None, relabel=False, recam=0):
        if 'query' in dir_path:
            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        else:
            img_paths = glob.glob(osp.join(dir_path, '*/*.jpg'))

        img_paths = sorted(img_paths)
        dataset = []
        for ii, img_path in enumerate(img_paths):
            pid = int(os.path.basename(img_path).split('_')[0])
            camid = int(recam + ii)
            if relabel and pid2label is not None:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid, 1))

        return dataset


    def print_dataset_statistics_movie(self, train, query, gallery, query_test, gallery_test):
        num_train_pids, num_train_imgs, num_train_cams, _ = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, _ = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, _ = self.get_imagedata_info(gallery)
        num_query_test_pids, num_query_test_imgs, num_query_test_cams, _ = self.get_imagedata_info(query_test)
        num_gallery_test_pids, num_gallery_test_imgs, num_gallery_test_cams, _ = self.get_imagedata_info(gallery_test)

        print("Dataset statistics:")
        print("  --------------------------------------")
        print("  subset         | # ids     | # images")
        print("  --------------------------------------")
        print("  train          | {:5d}     | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query          | {:5d}     | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery        | {:5d}     | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  query_test     | {:5d}     | {:8d}".format(num_query_test_pids, num_query_test_imgs))
        print("  gallery_test   | {:5d}     | {:8d}".format(num_gallery_test_pids, num_gallery_test_imgs))
























