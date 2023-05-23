import os
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import os.path as osp
import random
import torch
import torchvision
ImageFile.LOAD_TRUNCATED_IMAGES = True
import librosa
import torchaudio
import mxnet as mx
import numbers
import torch.nn.functional as F
import torchvision.transforms as T
from model.backbones.tokenization_bert import BertTokenizer
import re
import copy


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


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

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, trackid, img_path.split('/')[-1]


class ImageDatasetPRE(Dataset):
    def __init__(self, root, dataset, transform=None):
        self.root = root
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, caption = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)        # [3, 256, 128]

        return img, caption, pid, img_path


class ImageDatasetUVTLevel(Dataset):
    def __init__(self, root, dataset, transform=None):
        self.root = root
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path_, pid, caption, atts = self.dataset[index]
        img_path = os.path.join(self.root, img_path_)
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)        # [3, 256, 128]

        caption = caption.lower()
        if len(atts) > 0:
            key_list = list(atts.keys())
            num = random.randint(1, len(key_list))
            key_sel = random.sample(key_list, num)
            att_prc = [atts[item] for item in key_sel]
            att_prc = ",".join(att_prc)
        else:
            cap_list = self.split_txt(caption)
            att_prc = random.sample(cap_list, 1)[0]

        if len(att_prc) == 0:
            att_prc = caption

        return img, caption, pid, img_path, att_prc

    def split_txt(self, caption):
        cap_list = caption.strip('.').split(".")
        result = []
        for cap in cap_list:
            caps = cap.split(',')
            nums = [len(item) for item in caps]
            idx = np.argmax(nums)
            item_1 = caps[idx]
            item_2 = (",".join(caps[:idx]) + "," + ",".join(caps[idx + 1:])).strip(",")
            item = [item_1, item_2]
            item = [it for it in item if len(it) > 5]
            result += item
        result = [it.strip(" ") for it in result if len(it) > 0]
        return result


class ImageDatasetLevelData(Dataset):
    def __init__(self, root, dataset, transform=None):
        self.root = root
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path_, pid, caption, atts = self.dataset[index]
        img_path = os.path.join(self.root, img_path_)
        img = read_image(img_path)

        if self.transform is not None:
            img1 = self.transform(img)        # [3, 256, 128]
            img2 = self.transform(img)  # [3, 256, 128]
            img3 = self.transform(img)  # [3, 256, 128]

        caption = caption.lower()
        if len(atts) > 0:
            key_list = list(atts.keys())
            num = random.randint(1, len(key_list))
            key_sel = random.sample(key_list, num)
            att_prc = [atts[item] for item in key_sel]
            att_prc = ",".join(att_prc)
        else:
            cap_list = self.split_txt(caption)
            att_prc = random.sample(cap_list, 1)[0]

        if len(att_prc) == 0:
            att_prc = caption

        return img1, img2, img3, caption, pid, img_path, att_prc

    def split_txt(self, caption):
        cap_list = caption.strip('.').split(".")
        result = []
        for cap in cap_list:
            caps = cap.split(',')
            nums = [len(item) for item in caps]
            idx = np.argmax(nums)
            item_1 = caps[idx]
            item_2 = (",".join(caps[:idx]) + "," + ",".join(caps[idx + 1:])).strip(",")
            item = [item_1, item_2]
            item = [it for it in item if len(it) > 5]
            result += item
        result = [it.strip(" ") for it in result if len(it) > 0]
        return result



class ImageDatasetLevelICFG(Dataset):
    def __init__(self, root, dataset, transform=None):
        self.root = root
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path_, pid, caption, processed_tokens = self.dataset[index]
        img_path = os.path.join(self.root, img_path_)
        img = read_image(img_path)

        if self.transform is not None:
            img1 = self.transform(img)        # [3, 256, 128]
            img2 = self.transform(img)  # [3, 256, 128]
            img3 = self.transform(img)  # [3, 256, 128]

        caption = caption.lower()
        cap_list = caption.strip('.').strip('[').strip(']').split(".")
        result = []
        for cap in cap_list:
            caps = cap.split(',')
            item = [it for it in caps if len(it) > 5]
            result += item
        result = [it.strip(" ") for it in result if len(it) > 0]
        word = random.sample(result, 1)[0]

        return img1, img2, img3, caption, pid, img_path, word




class ImageDatasetUVT(Dataset):
    def __init__(self, root, dataset, transform=None):
        self.root = root
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path_, pid, caption, token = self.dataset[index]
        img_path = os.path.join(self.root, img_path_)
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)        # [3, 256, 128]

        return img, caption, pid, img_path, token




class ImageDatasetLevelRSTP(Dataset):
    def __init__(self, root, dataset, transform=None):
        self.root = root
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path_, pid, caption, processed_tokens = self.dataset[index]
        img_path = os.path.join(self.root, img_path_)
        img = read_image(img_path)

        if self.transform is not None:
            img1 = self.transform(img)        # [3, 256, 128]
            img2 = self.transform(img)  # [3, 256, 128]
            img3 = self.transform(img)  # [3, 256, 128]

        caption = caption.lower()
        cap_list = caption.strip('.').strip('[').strip(']').split(".")
        result = []
        for cap in cap_list:
            caps = cap.split(',')
            item = [it for it in caps if len(it) > 5]
            result += item
        result = [it.strip(" ") for it in result if len(it) > 0]
        word = random.sample(result, 1)[0]

        return img1, img2, img3, caption, pid, img_path, word


from random import randint, shuffle
from random import random as rand


class TextMaskingGenerator:
    def __init__(self, tokenizer, mask_prob, mask_max, skipgram_prb=0.2, skipgram_size=3, mask_whole_word=True, use_roberta=False):
        self.id2token = {i: w for w, i in tokenizer.get_vocab().items()}
        print("len(tokenizer.id2token), ", len(self.id2token), flush=True)

        self.use_roberta = use_roberta

        for i in range(len(self.id2token)):
            assert i in self.id2token.keys()  # check

        self.cls_token = tokenizer.cls_token
        self.mask_token = tokenizer.mask_token
        print("mask_generator.cls_token, ", self.cls_token, flush=True)
        print("mask_generator.mask_token, ", self.mask_token, flush=True)

        self.mask_max = mask_max
        self.mask_prob = mask_prob

        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word

    def get_random_word(self):
        i = randint(0, len(self.id2token) - 1)
        return self.id2token[i]

    def __call__(self, tokens: list):  # tokens: [CLS] + ...
        n_pred = min(self.mask_max, max(
            1, int(round(len(tokens) * self.mask_prob))))

        # candidate positions of masked tokens
        assert tokens[0] == self.cls_token
        special_pos = set([0])  # will not be masked
        cand_pos = list(range(1, len(tokens)))

        shuffle(cand_pos)
        masked_pos = set()
        max_cand_pos = max(cand_pos)
        for pos in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            if pos in masked_pos:
                continue

            def _expand_whole_word(st, end):
                new_st, new_end = st, end

                if self.use_roberta:
                    while (new_st > 1) and (tokens[new_st][0] != 'Ġ'):
                        new_st -= 1
                    while (new_end < len(tokens)) and (tokens[new_end][0] != 'Ġ'):
                        new_end += 1
                else:
                    # bert, WordPiece
                    while (new_st >= 0) and tokens[new_st].startswith('##'):
                        new_st -= 1
                    while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
                        new_end += 1

                return new_st, new_end

            if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                # ngram
                cur_skipgram_size = randint(2, self.skipgram_size)
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(
                        pos, pos + cur_skipgram_size)
                else:
                    st_pos, end_pos = pos, pos + cur_skipgram_size
            else:
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                else:
                    st_pos, end_pos = pos, pos + 1

            for mp in range(st_pos, end_pos):
                if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                    masked_pos.add(mp)
                else:
                    break

        masked_pos = list(masked_pos)
        n_real_pred = len(masked_pos)
        if n_real_pred > n_pred:
            shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]

        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                tokens[pos] = self.mask_token
            elif rand() < 0.5:  # 10%
                tokens[pos] = self.get_random_word()

        return tokens, masked_pos


def pre_caption(caption, max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    if not len(caption):
        raise ValueError("pre_caption yields invalid text")

    return caption


class ImageDataset_IMGTXT_PRE(Dataset):
    def __init__(self, root, dataset, transform=None, max_words=50):
        self.root = root
        self.dataset = dataset
        self.transform = transform
        self.max_words = max_words
        self.max_tokens = 50
        self.max_masks = 8
        self.tokenizer = BertTokenizer.from_pretrained('bert_tool/uncased_L-12_H-768_A-12')

        self.cls_token = self.tokenizer.cls_token
        self.eos_token = self.tokenizer.sep_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        self.PAD_mask = -100

        self.mask_generator = TextMaskingGenerator(self.tokenizer, 0.25, 8, 0.2, 3, True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, caption = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)        # [3, 256, 128]

        text_ids, text_atts, text_ids_masked, masked_pos, masked_ids = self.preprocess(caption)

        return img, caption, pid, text_ids, text_atts, text_ids_masked, masked_pos, masked_ids, img_path

    def preprocess(self, text):
        # if self.tokenized:
        #     tokens = text.strip().split(' ')
        # else:
        text = pre_caption(text, self.max_words)  # be careful, if text is '', it will cause error
        tokens = self.tokenizer.tokenize(text)

        tokens = [self.cls_token] + tokens[:self.max_tokens - 1]

        # if self.add_eos:
        if False:
            tokens = tokens[:self.max_tokens - 1]
            tokens += [self.eos_token]

        n_tokens = len(tokens)
        assert n_tokens >= 2, "len(word tokens) < 2"

        text_ids = self.tokenizer.convert_tokens_to_ids(tokens)  # list of int

        tokens_masked, masked_pos = self.mask_generator(copy.deepcopy(tokens))
        text_ids_masked = self.tokenizer.convert_tokens_to_ids(tokens_masked)  # list of int
        masked_ids = [text_ids[p] for p in masked_pos]

        # pad
        n_pad = self.max_tokens - n_tokens
        text_ids = text_ids + [self.pad_token_id] * n_pad
        text_atts = [1] * n_tokens + [0] * n_pad

        text_ids_masked = text_ids_masked + [self.pad_token_id] * n_pad
        n_pad = self.max_masks - len(masked_ids)
        masked_pos = masked_pos + [0] * n_pad
        masked_ids = masked_ids + [self.PAD_mask] * n_pad

        return text_ids, text_atts, text_ids_masked, masked_pos, masked_ids


class ImageDatasetPath(Dataset):
    def __init__(self, root, dataset, transform=None):
        self.root = root
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path_, pid, cam, _ = self.dataset[index]
        img_path = os.path.join(self.root, img_path_)
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)        # [3, 256, 128]

        return img, pid, cam, img_path


class ImageDatasetLaST(Dataset):
    def __init__(self, root, dataset, transform=None):
        self.root = root
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, caption = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)        # [3, 256, 128]

        return img, caption, pid, img_path




