import os
import glob
import os.path as osp
import numpy as np
import random
import pickle
import json
import errno


def mkdir_if_missing(dir_path):
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(os.path.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))

def proc_data_json(id_item, id2info):
    result = []
    for id in id_item:
        info_list = id2info[id]
        for info in info_list:
            id_ = info['id']
            img_path = info['img_path']
            captions = info['captions']
            for cap in captions:
                result.append({'id': id_, 'img_path': img_path, 'captions': cap})
    return result


def get_id2info(data_json):
    id2info = {}
    for img in data_json:
        id = data_json[img]['id']
        try:
            id2info[id].append(data_json[img])
        except:
            id2info[id] = [data_json[img]]
    return id2info


home = r'/data/reid/ICFG_PEDES/ICFG_PEDES'
file = r'ICFG-PEDES.json'


path_f = os.path.join(home, file)
data_json = read_json(path_f)           # [54522,]

info_train = [item for item in data_json if item['split'] == 'train']           # [34674,]
info_test = [item for item in data_json if item['split'] == 'test']             # [19848,]


id_list = sorted(list(set([item['id'] for item in data_json])))                 # [4102,]
num_txt = [len(item['captions']) for item in data_json]                         # [54522,], min=1, max=1
captions = [item['captions'] for item in data_json]                             # [20505,]

id_train = list(set([item['id'] for item in info_train]))                       # [3102,]
id_test = list(set([item['id'] for item in info_test]))                         # [1000,]


result = {'train': info_train, 'test': info_test}
path_save = os.path.join(home, 'ICFG_PEDES_PROCESS.json')
write_json(result, path_save)

print('finish')




