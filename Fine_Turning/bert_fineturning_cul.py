# from watchmen import WatchClient
#
# client = WatchClient(id="short description of this running", gpus=[0],
#                      server_host="127.0.0.1", server_port=62333)
# client.wait()

import sys
sys.path.append("../")
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import json
import torch
import argparse
import torch.nn as nn
import torch.nn.utils as utils
from torch.utils.data import DataLoader
from Fine_Turning.metircs import Metrics
import logging
from torch.utils.data import RandomSampler, WeightedRandomSampler
from transformers import AdamW
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from models.tokenizer import Tokenizer
from models.tokenizer_visible_mask import Tokenizer as VMTokenizer
from models.custom_mask_model import CustomMaskModel
# from models.du_bert import DuBert
# from models.du_bert2 import DuBert as DuBert2
# from models.du_bert3 import DuBert as DuBert3
# from models.du_bert4 import DuBert as DuBert4
# from models.du_bert5 import DuBert as DuBert5
# from models.du_bert6 import DuBert as DuBert6
# from models.du_bert7 import DuBert as DuBert7
# from models.du_bert8 import DuBert as DuBert8
# from models.du_bert9 import DuBert as DuBert9
# from models.du_bert10 import DuBert as DuBert10
# from models.du_bert11 import DuBert as DuBert11
# from models.du_bert12 import DuBert as DuBert12
# from models.du_bert13 import DuBert as DuBert13
# from models.du_bert import DuBert
# from models.du_bert_unilm import DuBertUniLM
# from models.du_bert_pretrain import DuBertPretrain
# from models.du_bert_pretrain2 import DuBertPretrain as DuBertPretrain2
# from models.du_bert_bce import DuBert as DuBertBce
# from models.du_bert_bear import DuBert as DuBertBear
# from models.du_bert_distill import DuBertDistill
# from models.du_bert_weighted import DuBertWeighted
# from models.du_bert_wocl import DuBertWoCL
# from models.du_bert_extend import DuBertExtend
# from models.du_bert_character import DuBertCharacter
# from models.du_bert_cul import DuBertCul
# from models.du_bert_cul_margin import DuBertCulMargin
# from models.du_bert_fine_grained import DuBertFineGrained
# from models.du_bert_fine_grained41 import DuBertFineGrained as DuBertFineGrained41
# from models.du_bert_fine_grained42 import DuBertFineGrained as DuBertFineGrained42
# from models.du_bert_fine_grained43 import DuBertFineGrained as DuBertFineGrained43
# from models.du_bert_fine_grained44 import DuBertFineGrained as DuBertFineGrained44
# from models.du_bert_fine_grained5 import DuBertFineGrained as DuBertFineGrained5
# from models.du_bert_fine_grained6 import DuBertFineGrained as DuBertFineGrained6
# from models.du_bert_fine_grained7 import DuBertFineGrained as DuBertFineGrained7
# from models.du_bert_fine_grained8 import DuBertFineGrained as DuBertFineGrained8
# from models.du_bert_fine_grained9 import DuBertFineGrained as DuBertFineGrained9
# from models.du_bert_fine_grained10 import DuBertFineGrained as DuBertFineGrained10
# from models.du_bert_fine_grained11 import DuBertFineGrained as DuBertFineGrained11
# from models.du_bert_fine_grained12 import DuBertFineGrained as DuBertFineGrained12
# from models.du_bert_fine_grained13 import DuBertFineGrained as DuBertFineGrained13
# from models.du_bert_fine_grained14 import DuBertFineGrained as DuBertFineGrained14
# from models.du_bert_fine_grained15 import DuBertFineGrained as DuBertFineGrained15
# from models.du_bert_fine_grained16 import DuBertFineGrained as DuBertFineGrained16
# from models.du_bert_cross_layer import DuBertCrossLayer
# from models.du_bert_cross_layer2 import DuBertCrossLayer as DuBertCrossLayer2
# from models.du_bert_cross_layer3 import DuBertCrossLayer as DuBertCrossLayer3
# from models.du_bert_cross_layer4 import DuBertCrossLayer as DuBertCrossLayer4
# from models.du_bert_cross_layer5 import DuBertCrossLayer as DuBertCrossLayer5
# from models.du_bert_cross_layer6 import DuBertCrossLayer as DuBertCrossLayer6
# from models.du_bert_cross_layer7 import DuBertCrossLayer as DuBertCrossLayer7
# from models.du_bert_cross_layer8 import DuBertCrossLayer as DuBertCrossLayer8
# from models.du_bert_cross_layer9 import DuBertCrossLayer as DuBertCrossLayer9
# from models.du_bert_cross_layer10 import DuBertCrossLayer as DuBertCrossLayer10
# from models.du_bert_cross_layer11 import DuBertCrossLayer as DuBertCrossLayer11
# from models.du_bert_cross_layer12 import DuBertCrossLayer as DuBertCrossLayer12
# from models.du_bert_cross_layer13 import DuBertCrossLayer as DuBertCrossLayer13
# from models.du_bert_cross_layer14 import DuBertCrossLayer as DuBertCrossLayer14
# from models.du_bert_cross_layer15 import DuBertCrossLayer as DuBertCrossLayer15
# from models.du_bert_cross_layer16 import DuBertCrossLayer as DuBertCrossLayer16
# from models.du_bert_resample import DuBertResample as DuBertResample
# from models.du_bert_resample2 import DuBertResample as DuBertResample2
# from models.du_bert_resample3 import DuBertResample as DuBertResample3
# from models.du_bert_resample4 import DuBertResample as DuBertResample4
# from models.du_bert_resample5 import DuBertResample as DuBertResample5
# from models.du_bert_resample6 import DuBertResample as DuBertResample6
# from models.du_bert_resample7 import DuBertResample as DuBertResample7
# from models.du_bert_resample8 import DuBertResample as DuBertResample8
# from models.du_bert_resample9 import DuBertResample as DuBertResample9
# from models.du_bert_resample10 import DuBertResample as DuBertResample10
# from models.du_bert_resample11 import DuBertResample as DuBertResample11
# from models.du_bert_resample12 import DuBertResample as DuBertResample12
# from models.du_bert_dusoftmax import DuBertDuSoftmax
# from models.du_bert_dusoftmax2 import DuBertDuSoftmax as DuBertDuSoftmax2
# from models.du_bert_dusoftmax3 import DuBertDuSoftmax as DuBertDuSoftmax3
# from models.du_bert_dusoftmax4 import DuBertDuSoftmax as DuBertDuSoftmax4
# from models.du_bert_dusoftmax5 import DuBertDuSoftmax as DuBertDuSoftmax5
# from models.du_bert_dusoftmax6 import DuBertDuSoftmax as DuBertDuSoftmax6
# from models.du_bert_dusoftmax7 import DuBertDuSoftmax as DuBertDuSoftmax7
# from models.du_bert_dusoftmax8 import DuBertDuSoftmax as DuBertDuSoftmax8
# from models.du_bert_dusoftmax9 import DuBertDuSoftmax as DuBertDuSoftmax9
# from models.du_bert_dusoftmax10 import DuBertDuSoftmax as DuBertDuSoftmax10
# from models.du_bert_dusoftmax12 import DuBertDuSoftmax as DuBertDuSoftmax12
# from models.du_bert_simcse1 import DuBertSimCSE as DuBertSimCSE1
# from models.du_bert_simcse2 import DuBertSimCSE as DuBertSimCSE2
# from models.du_bert_simcse3 import DuBertSimCSE as DuBertSimCSE3
# from models.du_bert_selector import DuBertSelector
# from models.du_bert_selector1 import DuBertSelector as DuBertSelector1
# from models.du_bert_selector2 import DuBertSelector as DuBertSelector2
# from models.du_bert_selector3 import DuBertSelector as DuBertSelector3
# from models.du_bert_selector4 import DuBertSelector as DuBertSelector4
# from models.du_bert_selector5 import DuBertSelector as DuBertSelector5
# from models.du_bert_selector6 import DuBertSelector as DuBertSelector6
# from models.du_bert_selector7 import DuBertSelector as DuBertSelector7
# from models.du_bert_selector8 import DuBertSelector as DuBertSelector8
# from models.du_bert_selector9 import DuBertSelector as DuBertSelector9
# from models.du_bert_selector10 import DuBertSelector as DuBertSelector10
# from models.du_bert_selector11 import DuBertSelector as DuBertSelector11
# from models.du_bert_selector12 import DuBertSelector as DuBertSelector12
# from models.du_bert_selector13 import DuBertSelector as DuBertSelector13
# from models.du_bert_selector14 import DuBertSelector as DuBertSelector14
# from models.du_bert_selector15 import DuBertSelector as DuBertSelector15
# from models.du_bert_selector16 import DuBertSelector as DuBertSelector16
# from models.du_bert_selector17 import DuBertSelector as DuBertSelector17
# from models.du_bert_selector18 import DuBertSelector as DuBertSelector18
# from models.du_bert_selector19 import DuBertSelector as DuBertSelector19
# from models.du_bert_selector20 import DuBertSelector as DuBertSelector20
# from models.du_bert_selector21 import DuBertSelector as DuBertSelector21
# from models.du_bert_selector22 import DuBertSelector as DuBertSelector22
# from models.du_bert_selector23 import DuBertSelector as DuBertSelector23
#
# from models.du_bert_mix import DuBertMix
# from models.du_bert_mix0 import DuBertMix as DuBertMix0
# from models.du_bert_mix2 import DuBertMix as DuBertMix2
# from models.du_bert_mix3 import DuBertMix as DuBertMix3
# from models.du_bert_mix4 import DuBertMix as DuBertMix4
# from models.du_bert_mix5 import DuBertMix as DuBertMix5
# from models.du_bert_mix6 import DuBertMix as DuBertMix6
# from models.du_bert_mix7 import DuBertMix as DuBertMix7
#
# from models.du_bert_cache import DuBertCache
# from models.du_bert_cache0 import DuBertCache as DuBertCache0
#
# from models.du_bert_wti import DuBertWti
# from models.du_bert_wti0 import DuBertWti as DuBertWti0
# from models.du_bert_wti2 import DuBertWti as DuBertWti2
# from models.du_bert_wti3 import DuBertWti as DuBertWti3
# from models.du_bert_wti4 import DuBertWti as DuBertWti4
# from models.du_bert_wti5 import DuBertWti as DuBertWti5
# from models.du_bert_wti6 import DuBertWti as DuBertWti6
# from models.du_bert_wti7 import DuBertWti as DuBertWti7
# from models.du_bert_wti8 import DuBertWti as DuBertWti8
from models.poly_encoder import PolyEncoder
from models.du_bert_pretrain import DuBertPretrain
from tqdm import tqdm
from torch.utils.data import Dataset

import torch.nn.functional as F
import os
print("bert finturning.py: ", os.getcwd())
# FT_model = {
#     'ubuntu': 'bert-base-uncased',
#     'douban': 'bert-base-chinese',
#     'e_commerce': 'bert-base-chinese'
# }
FT_model = {
    'ubuntu': '../pretrain_models/bert-base-uncased',
    'douban': '../pretrain_models/bert-base-chinese',
    'e_commerce': '../pretrain_models/bert-base-chinese'
}
FT_data = {
    'ubuntu': '../data/ubuntu_data/tokenized_data.json',
    'douban': '../data/douban_data/tokenized_data.json',
    'e_commerce': '../data/e_commerce_data/tokenized_data.json'
}
MODEL_CLASSES = {
    'bert': (BertConfig, CustomMaskModel, VMTokenizer),
    'custom_mask_model': (BertConfig, CustomMaskModel, VMTokenizer),
    "poly_encoder": (BertConfig, PolyEncoder, Tokenizer),
    # 'du_bert': (BertConfig, DuBert, Tokenizer), # du_bert_cross_entropy
    # 'du_bert2': (BertConfig, DuBert2, Tokenizer),
    # 'du_bert3': (BertConfig, DuBert3, Tokenizer),
    # 'du_bert4': (BertConfig, DuBert4, Tokenizer),
    # 'du_bert5': (BertConfig, DuBert5, Tokenizer),
    # 'du_bert6': (BertConfig, DuBert6, Tokenizer),
    # 'du_bert7': (BertConfig, DuBert7, Tokenizer),
    # 'du_bert8': (BertConfig, DuBert8, Tokenizer),
    # 'du_bert9': (BertConfig, DuBert9, Tokenizer),
    # 'du_bert10': (BertConfig, DuBert10, Tokenizer),
    # 'du_bert11': (BertConfig, DuBert11, Tokenizer),
    # 'du_bert12': (BertConfig, DuBert12, Tokenizer),
    # 'du_bert13': (BertConfig, DuBert13, Tokenizer),
    # 'du_bert_pretrain': (BertConfig, DuBertPretrain, Tokenizer),
    # 'du_bert_pretrain2': (BertConfig, DuBertPretrain2, Tokenizer),
    # 'du_bert_bce': (BertConfig, DuBertBce, Tokenizer),
    # 'du_bert_bear': (BertConfig, DuBertBear, Tokenizer),
    # 'du_bert_weighted': (BertConfig, DuBertWeighted, Tokenizer),
    # 'du_bert_distill': (BertConfig, DuBertDistill, Tokenizer),
    # 'du_bert_wocl': (BertConfig, DuBertWoCL, Tokenizer),
    # 'du_bert_extend': (BertConfig, DuBertExtend, Tokenizer),
    # 'du_bert_character': (BertConfig, DuBertCharacter, Tokenizer),
    # 'du_bert_cul': (BertConfig, DuBertCul, Tokenizer),
    # 'du_bert_cul_margin': (BertConfig, DuBertCulMargin, Tokenizer),
    # 'du_bert_fine_grained': (BertConfig, DuBertFineGrained, Tokenizer),
    # 'du_bert_fine_grained41': (BertConfig, DuBertFineGrained41, Tokenizer),
    # 'du_bert_fine_grained42': (BertConfig, DuBertFineGrained42, Tokenizer),
    # 'du_bert_fine_grained43': (BertConfig, DuBertFineGrained43, Tokenizer),
    # 'du_bert_fine_grained44': (BertConfig, DuBertFineGrained44, Tokenizer),
    # 'du_bert_fine_grained5': (BertConfig, DuBertFineGrained5, Tokenizer),
    # 'du_bert_fine_grained6': (BertConfig, DuBertFineGrained6, Tokenizer),
    # 'du_bert_fine_grained7': (BertConfig, DuBertFineGrained7, Tokenizer),
    # 'du_bert_fine_grained8': (BertConfig, DuBertFineGrained8, Tokenizer),
    # 'du_bert_fine_grained9': (BertConfig, DuBertFineGrained9, Tokenizer),
    # 'du_bert_fine_grained10': (BertConfig, DuBertFineGrained10, Tokenizer),
    # 'du_bert_fine_grained11': (BertConfig, DuBertFineGrained11, Tokenizer),
    # 'du_bert_fine_grained12': (BertConfig, DuBertFineGrained12, Tokenizer),
    # 'du_bert_fine_grained13': (BertConfig, DuBertFineGrained13, Tokenizer),
    # 'du_bert_fine_grained14': (BertConfig, DuBertFineGrained14, Tokenizer),
    # 'du_bert_fine_grained15': (BertConfig, DuBertFineGrained15, Tokenizer),
    # 'du_bert_fine_grained16': (BertConfig, DuBertFineGrained16, Tokenizer),
    # 'du_bert_du_softmax': (BertConfig, DuBertDuSoftmax, Tokenizer),
    # 'du_bert_du_softmax2': (BertConfig, DuBertDuSoftmax2, Tokenizer),
    # 'du_bert_du_softmax3': (BertConfig, DuBertDuSoftmax3, Tokenizer),
    # 'du_bert_du_softmax4': (BertConfig, DuBertDuSoftmax4, Tokenizer),
    # 'du_bert_du_softmax5': (BertConfig, DuBertDuSoftmax5, Tokenizer),
    # 'du_bert_du_softmax6': (BertConfig, DuBertDuSoftmax6, Tokenizer),
    # 'du_bert_du_softmax7': (BertConfig, DuBertDuSoftmax7, Tokenizer),
    # 'du_bert_du_softmax8': (BertConfig, DuBertDuSoftmax8, Tokenizer),
    # 'du_bert_du_softmax9': (BertConfig, DuBertDuSoftmax9, Tokenizer),
    # 'du_bert_du_softmax10': (BertConfig, DuBertDuSoftmax10, Tokenizer),
    # 'du_bert_du_softmax12': (BertConfig, DuBertDuSoftmax12, Tokenizer),
    # 'du_bert_sim_cse1': (BertConfig, DuBertSimCSE1, Tokenizer),
    # 'du_bert_sim_cse2': (BertConfig, DuBertSimCSE2, Tokenizer),
    # 'du_bert_sim_cse3': (BertConfig, DuBertSimCSE3, Tokenizer),
    # 'du_bert_unilm': (BertConfig, DuBertUniLM, Tokenizer),
    # 'du_bert_cross_layer': (BertConfig, DuBertCrossLayer, Tokenizer),
    # 'du_bert_cross_layer2': (BertConfig, DuBertCrossLayer2, Tokenizer),
    # 'du_bert_cross_layer3': (BertConfig, DuBertCrossLayer3, Tokenizer),
    # 'du_bert_cross_layer4': (BertConfig, DuBertCrossLayer4, Tokenizer),
    # 'du_bert_cross_layer5': (BertConfig, DuBertCrossLayer5, Tokenizer),
    # 'du_bert_cross_layer6': (BertConfig, DuBertCrossLayer6, Tokenizer),
    # 'du_bert_cross_layer7': (BertConfig, DuBertCrossLayer7, Tokenizer),
    # 'du_bert_cross_layer8': (BertConfig, DuBertCrossLayer8, Tokenizer),
    # 'du_bert_cross_layer9': (BertConfig, DuBertCrossLayer9, Tokenizer),
    # 'du_bert_cross_layer10': (BertConfig, DuBertCrossLayer10, Tokenizer),
    # 'du_bert_cross_layer11': (BertConfig, DuBertCrossLayer11, Tokenizer),
    # 'du_bert_cross_layer12': (BertConfig, DuBertCrossLayer12, Tokenizer),
    # 'du_bert_cross_layer13': (BertConfig, DuBertCrossLayer13, Tokenizer),
    # 'du_bert_cross_layer14': (BertConfig, DuBertCrossLayer14, Tokenizer),
    # 'du_bert_cross_layer15': (BertConfig, DuBertCrossLayer15, Tokenizer),
    # 'du_bert_cross_layer16': (BertConfig, DuBertCrossLayer16, Tokenizer),
    # 'du_bert_resample': (BertConfig, DuBertResample, Tokenizer),
    # 'du_bert_resample2': (BertConfig, DuBertResample2, Tokenizer),
    # 'du_bert_resample3': (BertConfig, DuBertResample3, Tokenizer),
    # 'du_bert_resample4': (BertConfig, DuBertResample4, Tokenizer),
    # 'du_bert_resample5': (BertConfig, DuBertResample5, Tokenizer),
    # 'du_bert_resample6': (BertConfig, DuBertResample6, Tokenizer),
    # 'du_bert_resample7': (BertConfig, DuBertResample7, Tokenizer),
    # 'du_bert_resample8': (BertConfig, DuBertResample8, Tokenizer),
    # 'du_bert_resample9': (BertConfig, DuBertResample9, Tokenizer),
    # 'du_bert_resample10': (BertConfig, DuBertResample10, Tokenizer),
    # 'du_bert_resample11': (BertConfig, DuBertResample11, Tokenizer),
    # 'du_bert_resample12': (BertConfig, DuBertResample12, Tokenizer),
    # 'du_bert_selector': (BertConfig, DuBertSelector, Tokenizer),
    # 'du_bert_selector1': (BertConfig, DuBertSelector1, Tokenizer),
    # 'du_bert_selector2': (BertConfig, DuBertSelector2, Tokenizer),
    # 'du_bert_selector3': (BertConfig, DuBertSelector3, Tokenizer),
    # 'du_bert_selector4': (BertConfig, DuBertSelector4, Tokenizer),
    # 'du_bert_selector5': (BertConfig, DuBertSelector5, Tokenizer),
    # 'du_bert_selector6': (BertConfig, DuBertSelector6, Tokenizer),
    # 'du_bert_selector7': (BertConfig, DuBertSelector7, Tokenizer),
    # 'du_bert_selector8': (BertConfig, DuBertSelector8, Tokenizer),
    # 'du_bert_selector9': (BertConfig, DuBertSelector9, Tokenizer),
    # 'du_bert_selector10': (BertConfig, DuBertSelector10, Tokenizer),
    # 'du_bert_selector11': (BertConfig, DuBertSelector11, Tokenizer),
    # 'du_bert_selector12': (BertConfig, DuBertSelector12, Tokenizer),
    # 'du_bert_selector13': (BertConfig, DuBertSelector13, Tokenizer),
    # 'du_bert_selector14': (BertConfig, DuBertSelector14, Tokenizer),
    # 'du_bert_selector15': (BertConfig, DuBertSelector15, Tokenizer),
    # 'du_bert_selector16': (BertConfig, DuBertSelector16, Tokenizer),
    # 'du_bert_selector17': (BertConfig, DuBertSelector17, Tokenizer),
    # 'du_bert_selector18': (BertConfig, DuBertSelector18, Tokenizer),
    # 'du_bert_selector19': (BertConfig, DuBertSelector19, Tokenizer),
    # 'du_bert_selector20': (BertConfig, DuBertSelector20, Tokenizer),
    # 'du_bert_selector21': (BertConfig, DuBertSelector21, Tokenizer),
    # 'du_bert_selector22': (BertConfig, DuBertSelector22, Tokenizer),
    # 'du_bert_selector23': (BertConfig, DuBertSelector23, Tokenizer),
    #
    # 'du_bert_mix0': (BertConfig, DuBertMix0, Tokenizer),
    # 'du_bert_mix': (BertConfig, DuBertMix, Tokenizer),
    # 'du_bert_mix2': (BertConfig, DuBertMix2, Tokenizer),
    # 'du_bert_mix3': (BertConfig, DuBertMix3, Tokenizer),
    # 'du_bert_mix4': (BertConfig, DuBertMix4, Tokenizer),
    # 'du_bert_mix5': (BertConfig, DuBertMix5, Tokenizer),
    # 'du_bert_mix6': (BertConfig, DuBertMix6, Tokenizer),
    # 'du_bert_mix7': (BertConfig, DuBertMix7, Tokenizer),
    #
    # 'du_bert_cache': (BertConfig, DuBertCache, Tokenizer),
    # 'du_bert_cache0': (BertConfig, DuBertCache0, Tokenizer),
    # 'du_bert_wti0': (BertConfig, DuBertWti0, Tokenizer),
    # 'du_bert_wti': (BertConfig, DuBertWti, Tokenizer),
    # 'du_bert_wti2': (BertConfig, DuBertWti2, Tokenizer),
    # 'du_bert_wti3': (BertConfig, DuBertWti3, Tokenizer),
    # 'du_bert_wti4': (BertConfig, DuBertWti4, Tokenizer),
    # 'du_bert_wti5': (BertConfig, DuBertWti5, Tokenizer),
    # 'du_bert_wti6': (BertConfig, DuBertWti6, Tokenizer),
    # 'du_bert_wti7': (BertConfig, DuBertWti7, Tokenizer),
    # 'du_bert_wti8': (BertConfig, DuBertWti8, Tokenizer),

    'du_bert_pretrain': (BertConfig, DuBertPretrain, Tokenizer)
}

torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
logger = logging.getLogger(__name__)


class BERTDataset(Dataset):
    def __init__(self, args, train, tokenizer=None):
        self.train = train
        self.args = args
        self.bert_tokenizer = tokenizer

    def __len__(self):
        # return 200
        return len(self.train) // 2

    def __getitem__(self, item):
        # example = self.train[item]
        # return example['context'], example['response'], example['label']
        item = item * 2
        e1 = self.train[item]
        e2 = self.train[item+1]
        return [e1, e2, item]


class Collate_fn(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        print("collate fn for model class: ", self.args.model_class)

    def __call__(self, inputs):
        # [[e1, e2], [e1, e2], [e1, e2], [e1, e2]]
        contexts, responses, labels = [], [], []
        item_ids = []
        for e in inputs:
            contexts += [e[0]['context'], e[1]['context']]
            responses += [e[0]['response'], e[1]['response']]
            labels += [e[0]['label'], e[1]['label']]
            item_ids.append(e[2])

        # # contexts, responses, labels = list(contexts), list(responses), list(labels)
        # if "pretrain" in self.args.model_class:
        #     inputs = self.tokenizer.batch_encode_plus(contexts, responses,
        #                                               return_tensor='pt', type='pretrain')
        # else:
        inputs = self.tokenizer.batch_encode_plus(contexts, responses,
                                                  return_tensor='pt', type='fine-turning')

        if self.args.model_class in ['du_bert', 'du_bert_pretrain']:
            labels = torch.LongTensor(labels)
        else:
            labels = torch.FloatTensor(labels)

        inputs['labels'] = labels
        inputs['item_ids'] = item_ids
        return inputs


class NeuralNetwork(nn.Module):

    def __init__(self, args):
        super(NeuralNetwork, self).__init__()
        self.args = args
        self.patience = 0
        self.init_clip_max_norm = 5.0
        self.optimizer = None
        self.best_result = [0, 0, 0, 0, 0, 0]
        self.metrics = Metrics(self.args.score_file_path)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_class = args.model_class
        # self.model_class = 'custom_mask_model'
        print("Model Class:", self.model_class)
        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.model_class]

        self.bert_config = config_class.from_pretrained(FT_model[args.task], num_labels=1)

        # self.bert_config.mix_type = "mix_token"
        self.bert_config.mix_type = args.mix_type
        self.bert_config.alpha = args.alpha
        self.bert_config.cl_layer = args.cl_layer
        if self.args.model_class == "poly_encoder":
            self.bert_config.cl_layer = 12
        # self.bert_tokenizer = BertTokenizer.from_pretrained(FT_model[args.task], do_lower_case=args.do_lower_case)
        self.bert_tokenizer = tokenizer_class(FT_model[args.task])
        self.collate_fn = Collate_fn(self.args, self.bert_tokenizer)

        self.bert_model = model_class.from_pretrained(FT_model[args.task], config=self.bert_config)

        self.bert_model.resize_token_embeddings(self.bert_config.vocab_size+1)
        if self.args.load_fp:
            """You can load the post-trained checkpoint here."""
            if self.args.task == 'ubuntu':
                self.bert_model.bert.load_state_dict(state_dict=torch.load("../pretrain_models/post-train/ubuntu25/bert.pt"), strict=False)
                print("---load ../pretrain_models/post-train/ubuntu25/bert.pt ")
            if self.args.task == 'douban':
                self.bert_model.bert.load_state_dict(state_dict=torch.load("../pretrain_models/post-train/douban27/bert.pt"), strict=False)
                print("---load ../pretrain_models/post-train/douban27/bert.pt")
            if self.args.task == 'e_commerce':
                message = self.bert_model.bert.load_state_dict(state_dict=torch.load("../pretrain_models/post-train/e_commerce34/bert.pt"), strict=False)
                print("---load ../pretrain_models/post-train/e_commerce34/bert.pt")
                print(message)
        self.bert_model.resize_token_embeddings(len(self.bert_tokenizer))
        self.bert_model.config.vocab_size = len(self.bert_tokenizer)

        self.bert_model = self.resize_decoder(self.bert_model, len(self.bert_tokenizer))
        self.bert_model = self.reset_seq_relationship(self.bert_model)

        # self.bert_model.load_state_dict(state_dict=torch.load(
        #     "/home/zjy/yzf/BERT_Retrieval/Fine_Turning/FT_checkpoint/douban.135.pt"),
        #                                 strict=False)
        # print("load from 135 ckpt.")
        # print("load from /home/featurize/work/BERT_Retrieval/FPT/PT_checkpoint/e_commerce_14_epoch/pytorch_model.bin")
        if self.args.load_pretrain:
            if self.args.task == 'ubuntu':
                # model_path = "../FPT/PT_checkpoint/from_bert_ubuntu_4_epoch"
                # self.bert_model = model_class.from_pretrained(model_path)
                # model_path = "ubuntu.pretrain.from_bert.pt910"
                model_path = "../FPT/PT_checkpoint/from_bert_ubuntu_11_epoch"
                # self.load_model(model_path)
                self.bert_model = model_class.from_pretrained(model_path)
                # self.bert_model = model_class.from_pretrained("../FPT/ubuntu_1_epoch_73")
                # self.bert_model = model_class.from_pretrained("../FPT/ubuntu_2_epoch_base")
                print("load pretrain ubuntu model...", model_path)

            if self.args.task == 'douban':
                model_path = "../FPT/PT_checkpoint/douban_7_epoch"
                self.bert_model = model_class.from_pretrained(model_path)
                # self.bert_model = model_class.from_pretrained("../FPT/PT_checkpoint/douban_12_epoch/")
                print("load pretrain douban model...", model_path)

            if self.args.task == 'e_commerce':
                # model_path = "../FPT/PT_checkpoint/e_commerce_3_epoch"
                model_path = "../FPT/PT_checkpoint/ft2_e_commerce_4_epoch"
                self.bert_model = model_class.from_pretrained(model_path,
                                                              config=self.bert_config)
                print("load pretrain e-commerce model...", model_path)

        self.bert_model = self.bert_model.cuda()

        self.align_scores = {}

    def reset_seq_relationship(self, bert_model):
        if not hasattr(bert_model, 'cls'): return bert_model
        if not hasattr(bert_model.cls, 'seq_relationship'): return bert_model
        bert_model.cls.seq_relationship = nn.Linear(bert_model.config.hidden_size, 2)
        print("-reset seq relationship module...")
        return bert_model

    def resize_decoder(self, bert_model, vocab_size, reset=False):
        if not hasattr(bert_model, 'cls'): return bert_model
        if not hasattr(bert_model.cls, 'predictions'): return bert_model
        if reset == True:
            bert_model.cls.predictions.decoder = nn.Linear(self.bert_model.config.hidden_size, vocab_size)
            print("-reset MLM module...")
            return bert_model

        decoder = bert_model.cls.predictions.decoder
        weight = decoder.weight.data
        bias = decoder.bias.data
        dtype = weight.dtype
        old_vocab_size, embedding_dim = weight.shape
        if old_vocab_size == vocab_size: return bert_model
        new_bias = torch.zeros(vocab_size, dtype=dtype)
        new_bias[:old_vocab_size] = bias

        new_weight = torch.normal(mean=torch.zeros(vocab_size, embedding_dim, dtype=dtype), std=0.02)
        new_weight[:old_vocab_size, :] = weight

        bert_model.cls.predictions.decoder.weight.data = new_weight
        bert_model.cls.predictions.decoder.bias.data = new_bias
        print("-resize decoder, vocab size: ", vocab_size)
        return bert_model

    def forward(self):
        raise NotImplementedError

    def to_device(self, data):
        device_data = {}
        for key, value in data.items():
            device_data[key] = value.cuda(device=self.device)
        return device_data


    def forward_step(self, data):
        if self.model_class not in ['bert', 'custom_mask_model']:
            output = self.bert_model(**data)
            return output
        else:
            input_ids = data['input_ids']
            attention_mask = data['attention_mask']
            token_type_ids = data['token_type_ids']
            if self.model_class == 'custom_mask_model':
                visible_mask = data['visible_mask']
                character_ids = data['character_ids']
            else:
                visible_mask = None
                character_ids = None
            output = self.bert_model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     visible_mask=visible_mask,
                                     character_ids=character_ids,
                                     labels=data['labels'])
            return output

    def train_step(self, i, data, threshold=None):
        item_ids = data.pop('item_ids')
        with torch.no_grad():
            data = self.to_device(data)
            # batch_ids, batch_mask, batch_seg, batch_y, batch_len = (item.cuda(device=self.device) for item in data)
        self.optimizer.zero_grad()
        if "unilm" in self.args.model_class:
            output = self.bert_model(**data, do_unilm=True)
            loss = output[1]
            loss.backward()
            self.optimizer.step()

        self.optimizer.zero_grad()
        labels = data['labels']
        if 'cul' in self.args.model_class:
            output = self.bert_model(**data, neg_threshold=threshold)
        else:
            output = self.bert_model(**data)

        loss = output[1]
        loss.backward()
        self.optimizer.step()
        # align_score = output[2]
        # for idx, score in zip(item_ids, align_score):
        #     self.align_scores[idx] = score
        if i % 5000 == 0:
            print('Batch[{}] - loss: {:.6f}  batch_size:{}'.format(i, loss.item(),
                                                                   labels.size(0)))
            if self.args.model_class == 'du_bert_fine_grained':
                print('Batch[{}] - fine grained loss: {:.6f}  batch_size:{}'.format(i, output[2].item(),
                                                                       labels.size(0)))
        return loss.cpu().detach()

    def fit(self, train, dev, test):
        if torch.cuda.is_available(): self.cuda()

        dataset = BERTDataset(self.args, train, self.bert_tokenizer)
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, sampler=sampler, num_workers=1, collate_fn=self.collate_fn)

        self.loss_func = nn.BCELoss()
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, correct_bias=True)
        # self.evaluate(dev)
        # threshold_step = len(dataloader) * self.args.epochs * 0.6
        threshold_step = len(dataloader) * self.args.epochs * 0.5
        threshold_step = int(threshold_step)
        global_step = 0
        threshold_base = 0.3
        self.evaluate(test, is_test=True, save_model=True)
        for epoch in range(self.args.epochs):
            print("\nEpoch ", epoch + 1, "/", self.args.epochs)
            avg_loss = 0
            self.train()
            # if epoch >= 2:
            #     self.args.model_class = 'du_bert'
            #     self.collate_fn = Collate_fn(self.args, self.bert_tokenizer)
            #     dataloader = DataLoader(dataset, batch_size=self.args.batch_size, sampler=sampler, num_workers=1,
            #                             collate_fn=self.collate_fn)

            for i, data in enumerate(tqdm(dataloader)):
                self.train()
                # if epoch >= 2 and self.patience >= 3:
                #     print("Reload the best model...")
                #     self.load_state_dict(torch.load(self.args.save_path))
                #     self.adjust_learning_rate()
                #     self.patience = 0
                if self.args.use_cul_neg:
                    threshold = threshold_base + (global_step / threshold_step) * (1-threshold_base)
                    threshold = min(threshold, 1.0)
                else:
                    threshold = None
                loss = self.train_step(i, data, threshold)
                global_step += 1
                if self.init_clip_max_norm is not None:
                    utils.clip_grad_norm_(self.parameters(), max_norm=self.init_clip_max_norm)
                avg_loss += loss.item()
                eval_step = 5000 if self.args.task in ("douban", "e_commerce") else 7000
                if i % eval_step == 0 and i != 0:
                    self.evaluate(dev, save_model=False)
                    self.evaluate(test, is_test=True, save_model=True)
                #     self.train()
                    # print('temp val = ', self.bert_model.temp.item())

            # if self.args.use_cul_pos:
            #     assert len(self.align_scores) != 0
            #     weight = [self.align_scores[i] for i in range(len(dataset))]
            #     sampler = WeightedRandomSampler(weight, num_samples=len(dataset), replacement=False)
            #     dataloader = DataLoader(dataset, batch_size=self.args.batch_size, sampler=sampler, num_workers=1, collate_fn=self.collate_fn)

            cnt = len(train) // self.args.batch_size + 1
            print("Average loss:{:.6f} ".format(avg_loss / cnt))
            self.evaluate(dev, save_model=False)
            self.evaluate(test, is_test=True, save_model=True)

    def adjust_learning_rate(self, decay_rate=.5):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
            self.args.learning_rate = param_group['lr']
        print("Decay learning rate to: ", self.args.learning_rate)

    def predict_align_score(self, train):
        dataset = BERTDataset(self.args, train, self.bert_tokenizer)
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, sampler=sampler, num_workers=1,
                                collate_fn=self.collate_fn)
        self.eval()
        for data in dataloader:
            item_ids = data.pop('item_ids')
            with torch.no_grad():
                data = self.to_device(data)
                align_score = self.bert_model.forward_align_score(**data)
                for idx, score in zip(item_ids, align_score):
                    self.align_scores[idx] = align_score

    def evaluate(self, dev, is_test=False, save_model=True):
        self.eval()
        y_pred = self.predict(dev)
        labels = [example['label'] for example in dev]
        with open(self.args.score_file_path, 'w') as output:
            for score, label in zip(y_pred, labels):
                output.write(
                    str(score) + '\t' +
                    str(label) + '\n'
                )
        if is_test == False and self.args.task != 'ubuntu':
            self.metrics.segment = 2
        else:
            self.metrics.segment = 10
        result = self.metrics.evaluate_all_metrics()
        print("Evaluation Result: \n",
              "MAP:", result[0], "\t",
              "MRR:", result[1], "\t",
              "P@1:", result[2], "\t",
              "R1:", result[3], "\t",
              "R2:", result[4], "\t",
              "R5:", result[5])

        if not is_test and result[3] + result[4] + result[5] > self.best_result[3] + self.best_result[4] + \
                self.best_result[5] and save_model:
            print("Best Result: \n",
                  "MAP:", self.best_result[0], "\t",
                  "MRR:", self.best_result[1], "\t",
                  "P@1:", self.best_result[2], "\t",
                  "R1:", self.best_result[3], "\t",
                  "R2:", self.best_result[4], "\t",
                  "R5:", self.best_result[5])
            self.patience = 0
            self.best_result = result
            torch.save(self.state_dict(), self.args.save_path)
            print("save model!!!\n")
        else:
            self.patience += 1

    def predict(self, dev):
        y_pred = []
        print("dev data len:", len(dev))
        dataset = BERTDataset(self.args, dev, self.bert_tokenizer)
        raw_count, select_count = 0, 0
        dataloader = DataLoader(dataset, batch_size=50, collate_fn=self.collate_fn, shuffle=False)
        for i, data in enumerate(dataloader):
            with torch.no_grad():
                data.pop('item_ids')
                data = self.to_device(data)
                labels = data['labels']
                if self.args.model_class == "poly_encoder":
                    data["labels"] = None
                # selected_output = self.bert_model(**data, random_select=True)
                output = self.bert_model(**data)
                # print("selected output reward = {}, raw output reward = {}".format(selected_output[1].item(), output[1].item()))
                # logits = torch.sigmoid(output[0]).squeeze(-1)
                # if output[1] > selected_output[1]:
                #     logits = output[0]
                #     raw_count += 1
                # else:
                #     logits = selected_output[0]
                #     select_count += 1
                logits = output[0]
                y_pred += logits
                # y_pred += logits.tolist()
            if i % 200 == 0:
                print('Batch[{}] batch_size:{}'.format(i, labels.size(0)))
        # print("select count = {}, raw_count = {}".format(select_count, raw_count))
        print("pred len: ", len(y_pred))
        return y_pred

    def load_model(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict=state_dict, strict=True)
        print("---load weight from ", path, "...")
        if torch.cuda.is_available(): self.cuda()


parser = argparse.ArgumentParser()
parser.add_argument("--task",
                    default='e_commerce',
                    type=str,
                    help="The dataset used for training and test.")

parser.add_argument("--model_class",
                    default='du_bert_weighted',
                    type=str,
                    help="Model Class")

parser.add_argument("--save_version",
                    default='11',
                    type=str)

parser.add_argument("--is_training",
                    action='store_true',
                    help="Training model or testing model?")

parser.add_argument("--batch_size",
                    default=12,
                    type=int,
                    help="The batch size.")

parser.add_argument("--learning_rate",
                    default=3e-5,
                    type=float,
                    help="The initial learning rate for Adamw.")

parser.add_argument("--epochs",
                    default=10,
                    type=int,
                    help="Total number of training epochs to perform.")

parser.add_argument("--save_path",
                    default="./FT_checkpoint/",
                    type=str,
                    help="The path to save model.")

parser.add_argument("--score_file_path",
                    default="./scorefile.txt",
                    type=str,
                    help="The path to save model.")

parser.add_argument("--do_lower_case", action='store_true', default=True,
                    help="Set this flag if you are using an uncased model.")

parser.add_argument("--max_sequence_length", default=256, type=int, help="max length of sequence")

parser.add_argument("--use_cul_pos", action='store_true')
parser.add_argument("--use_cul_neg", action='store_true')

parser.add_argument("--mix_type", default="mix_token", type=str)
parser.add_argument("--alpha", default=0.2, type=float)
parser.add_argument("--cl_layer", default=6, type=int)
parser.add_argument("--load_pretrain", action='store_true')
parser.add_argument("--load_fp", action='store_true')

args = parser.parse_args()
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

args.save_path += args.task + '.' + "{}.pt".format(args.save_version)
args.score_file_path = args.score_file_path
# load bert

print(args)
print("Task: ", args.task)


def load_json(file_name):
    with open(file_name, 'r', encoding='utf8') as f:
        data = json.load(f)
    return data


def train_model(train, dev):
    model = NeuralNetwork(args=args)
    model.fit(train, dev)


def test_model(test):
    model = NeuralNetwork(args=args)
    model.load_model(args.save_path)
    model.evaluate(test, is_test=True)

def eval_model(dev):
    model = NeuralNetwork(args=args)
    model.load_model(args.save_path)
    model.evaluate(dev, is_test=False)

if __name__ == '__main__':
    start = time.time()
    data = load_json(FT_data[args.task])
    train, dev, test = data['train'], data['dev'], data['test']
    # with open(FT_data[args.task], 'rb') as f:
    #     train, dev, test = pickle.load(f, encoding='ISO-8859-1')
    print('data loaded ... ...')
    if args.is_training==True:
        model = NeuralNetwork(args=args)
        model.fit(train, dev, test)
        test_model(test)
    else:
        eval_model(dev)
        test_model(test)

    end = time.time()
    print("use time: ", (end - start) / 60, " min")