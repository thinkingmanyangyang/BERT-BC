# for douban context len=2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append("../")
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import time
import logging
import argparse
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer, BertConfig
from transformers import BertForPreTraining
from models.du_bert_pretrain import DuBertPretrain
from models.du_bert_pretrain2 import DuBertPretrain as DuBertPretrain2
from models.du_bert_pretrain3 import DuBertPretrain as DuBertPretrain3
from models.du_bert_pretrain4 import DuBertPretrain as DuBertPretrain4
from models.du_bert_pretrain5 import DuBertPretrain as DuBertPretrain5
from models.du_bert import DuBert
from Fine_Turning.metircs import Metrics
from models.tokenizer import Tokenizer
from models.tokenizer_visible_mask import Tokenizer as VMTokenizer
from transformers import AdamW
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import pickle

from torch.utils.data import Dataset
import random
# from setproctitle import setproctitle
#
# setproctitle('(janghoon) e-commerce_final')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

FP_data = {
    'ubuntu': '../data/ubuntu_data/ubuntu_tokenized_post_train.json',
    'douban': '../data/douban_data/douban_tokenized_post_train.json',
    'e_commerce': '../data/e_commerce_data/e_commerce_tokenized_post_train.json'
}

FT_data = {
    'ubuntu': '../data/ubuntu_data/tokenized_data.json',
    'douban': '../data/douban_data/tokenized_data.json',
    'e_commerce': '../data/e_commerce_data/tokenized_data.json'
}

FT_model = {
    'ubuntu': '../pretrain_models/bert-base-uncased',
    'douban': '../pretrain_models/bert-base-chinese',
    'e_commerce': '../pretrain_models/bert-base-chinese'
}

MODEL_CLASSES = {
    'du_bert': (BertConfig, DuBert, Tokenizer), # du_bert_cross_entropy
    'du_bert_pretrain': (BertConfig, DuBertPretrain, Tokenizer),
    'du_bert_pretrain2': (BertConfig, DuBertPretrain2, Tokenizer),
    'du_bert_pretrain3': (BertConfig, DuBertPretrain3, Tokenizer),
    'du_bert_pretrain4': (BertConfig, DuBertPretrain4, Tokenizer),
    'du_bert_pretrain5': (BertConfig, DuBertPretrain5, Tokenizer),
    'bert': (BertConfig, BertForPreTraining, VMTokenizer)
    # 'du_bert_cul': (BertConfig, DuBertCul, Tokenizer),
    # 'du_bert_cul_margin': (BertConfig, DuBertCulMargin, Tokenizer)
}

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0

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
        return [e1, e2]

class PretrainDataset(Dataset):
    def __init__(self, task):
        self.sample_to_doc = []  # map sample index to doc and line
        # load samples into memory
        self.all_docs = []
        doc = []
        data_path = FP_data[task]
        with open(data_path, encoding='utf8') as f:
            crsets = json.load(f)
        # crsets = pickle.load(file=open(corpus_path, 'rb'))
        # crsets=crsets[:50000]#crsets[:50000]+crsets[500000:]
        cnt = 0
        lcnt = 0
        for crset in tqdm(crsets):
            for line in crset:
                if len(line.strip()) == 0:
                    cnt += 1
                    continue
                if len(line) < 10:
                    if len(line.split()) == 0:
                        cnt += 1
                        continue
                sample = {"doc_id": len(self.all_docs),
                          "line": len(doc),
                          "end": 0,
                          "linenum": 1
                          }
                self.sample_to_doc.append(sample)
                # if len(self.tokenizer.tokenize(line)) == 0:
                # print("여기")
                doc.append(line)

            if (len(doc) != 0):
                self.all_docs.append(doc)
            else:
                print("empty")
            if (len(doc) < 3):
                for i in range(len(doc) - 1):
                    self.sample_to_doc.pop()
                self.sample_to_doc[-1]['end'] = len(doc)
                lcnt += 1
            else:
                self.sample_to_doc.pop()
                self.sample_to_doc.pop()
                # self.sample_to_doc.pop()

            doc = []

        print(cnt, lcnt)

        for doc in self.all_docs:
            if len(doc) == 0:
                print("problem")
        # import json
        with open("sampled_data.json", 'w', encoding="utf8") as f:
            json.dump(self.sample_to_doc, f, indent=4, ensure_ascii=False)

    def __len__(self):
        return len(self.sample_to_doc)

    def __getitem__(self, item):

        sample = self.sample_to_doc[item]
        # 방법에 비해 문장 길이가 짧은경우.
        length = sample['end']
        if length != 0:
            context = []
            for i in range(length - 1):
                # tokens_a += self.tokenizer.tokenize(self.all_docs[sample["doc_id"]][i]) + [self.tokenizer.eos_token]
                context.append(self.all_docs[sample["doc_id"]][i])
            # 选择回复
            # 正回复
            pos_response = self.all_docs[sample["doc_id"]][length - 1]
            # 负回复
            neg_response = self.get_random_line(sample)
        else:
            sample = self.sample_to_doc[item]
            t1 = self.all_docs[sample["doc_id"]][sample["line"]]
            t2 = self.all_docs[sample["doc_id"]][sample["line"] + 1]
            # t3 = self.all_docs[sample["doc_id"]][sample["line"] + 2]
            context = [t1, t2]
            pre_context = []
            for i in range(0, sample['line']):
                utt = self.all_docs[sample["doc_id"]][i]
                pre_context.append(utt)
            rand = random.random()
            # if rand > 0.5:
            context = pre_context + context
            # print(context)
            # 选择回复
            # 正回复
            pos_response = self.all_docs[sample["doc_id"]][sample["line"] + 2]
            # 负回复
            rand = random.random()
            # 从同一个对话中选择一个作为负样本
            if rand > 0.85:
                samedoc = self.all_docs[sample["doc_id"]]
                linenum = random.randrange(len(samedoc))
                while linenum == sample["line"] + 1 or linenum == sample["line"] + 2:
                    linenum = random.randrange(len(samedoc))
                neg_response = samedoc[linenum]
            # 从其他对话中选择一个作为负样本
            else:
                neg_response = self.get_random_line(sample)

        assert len(pos_response) > 0
        assert len(neg_response) > 0
        pos_example = {
            "context": context,
            "response": pos_response,
            "label": 1
        }
        neg_example = {
            "context": context,
            "response": neg_response,
            "label": 0
        }
        # print("context", context)
        # print("pos_response", pos_response)
        # print("neg_response", neg_response)
        return pos_example, neg_example

    def get_random_line(self, sample):
        while (True):
            rand_doc_idx = random.randint(0, len(self.all_docs) - 1)
            if sample["doc_id"] != rand_doc_idx:
                break

        rand_doc = self.all_docs[rand_doc_idx]
        line = rand_doc[random.randrange(len(rand_doc))]
        return line


class Collate_fn(object):
    def __init__(self, tokenizer, mlm=True):
        self.tokenizer = tokenizer
        self.mlm = mlm

    def __call__(self, inputs):
        # [[e1, e2], [e1, e2], [e1, e2], [e1, e2]]
        contexts, responses, labels = [], [], []
        for e in inputs:
            contexts += [e[0]['context'], e[1]['context']]
            responses += [e[0]['response'], e[1]['response']]
            labels += [e[0]['label'], e[1]['label']]

        # contexts, responses, labels = list(contexts), list(responses), list(labels)
        if self.mlm:
            inputs = self.tokenizer.batch_encode_plus(contexts, responses,
                                                    return_tensor='pt', type='pretrain')
        else:
            inputs = self.tokenizer.batch_encode_plus(contexts, responses,
                                                    return_tensor='pt', type='fine-turning')
        labels = torch.LongTensor(labels)
        inputs['labels'] = labels
        return inputs

# dataset = PretrainDataset('e_commerce')
# print(len("dataset"))
# tokenizer = Tokenizer(FT_model['e_commerce'])
# # from argparse import Namespace
#
# collate_fn = Collate_fn(tokenizer)
# dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
# from tqdm import tqdm
# while True:
#     for item in tqdm(dataset):
#         pass
#     # print(item)
# exit()

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

        print("Model Class:", self.model_class)
        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.model_class]

        self.bert_config = config_class.from_pretrained(FT_model[args.task], num_labels=1)
        self.bert_config.cl_layer = args.cl_layer
        # self.bert_tokenizer = BertTokenizer.from_pretrained(FT_model[args.task], do_lower_case=args.do_lower_case)
        self.bert_tokenizer = tokenizer_class(FT_model[args.task])
        self.collate_fn = Collate_fn(self.bert_tokenizer, mlm=True)

        self.bert_model = model_class.from_pretrained(FT_model[args.task], config=self.bert_config)

        self.bert_model.resize_token_embeddings(self.bert_config.vocab_size+1)
        # """You can load the post-trained checkpoint here."""
        if self.args.load_fp:
            if self.args.task == 'ubuntu':
                self.bert_model.bert.load_state_dict(state_dict=torch.load("../pretrain_models/post-train/ubuntu25/bert.pt"), strict=False)
                print("---load ../pretrain_models/post-train/ubuntu25/bert.pt ")
            if self.args.task == 'douban':
                self.bert_model.bert.load_state_dict(state_dict=torch.load("../pretrain_models/post-train/douban27/bert.pt"), strict=False)
                print("---load ../pretrain_models/post-train/douban27/bert.pt")
            if self.args.task == 'e_commerce':
                self.bert_model.bert.load_state_dict(state_dict=torch.load("../pretrain_models/post-train/e_commerce34/bert.pt"), strict=False)
                print("---load ../pretrain_models/post-train/e_commerce34/bert.pt")

        self.bert_model.resize_token_embeddings(len(self.bert_tokenizer))
        self.bert_model.config.vocab_size = len(self.bert_tokenizer)
        self.bert_model = self.resize_decoder(self.bert_model, len(self.bert_tokenizer))
        self.bert_model = self.reset_seq_relationship(self.bert_model)
        # self.load_model("./FT_checkpoint/e_commerce.34.pt")
        # self.bert_model = model_class.from_pretrained("../FPT/ubuntu_3_epoch")
        # print("load ubuntu 3 epoch model ... ...")
        self.bert_model = self.bert_model.cuda()

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
        output = self.bert_model(**data)
        return output

    def train_step(self, i, data, threshold=None):
        data = self.to_device(data)
        self.optimizer.zero_grad()
        labels = data['labels']
        if 'cul' in self.args.model_class:
            output = self.bert_model(**data, neg_threshold=threshold)
        else:
            output = self.bert_model(**data)
        loss = output[1]
        loss.backward()
        self.optimizer.step()
        if i % 5000 == 0:
            print('Batch[{}] - loss: {:.6f}  batch_size:{}'.format(i, loss.item(),
                                                                   labels.size(0)))
        return loss.cpu().detach()

    def fit(self, train, dev, test):
        if torch.cuda.is_available(): self.cuda()

        dataset = PretrainDataset(self.args.task)
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, sampler=sampler, num_workers=1,
                                collate_fn=self.collate_fn, drop_last=True)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, correct_bias=True)
        # self.evaluate(dev)
        threshold_step = len(dataloader) * (self.args.epochs-2)
        global_step = 0
        threshold_base = 0.3
        before = 100.0
        for epoch in range(self.args.epochs):

            print("\nEpoch ", epoch + 1, "/", self.args.epochs)
            print("epoch learning rate is: ", self.args.learning_rate)
            avg_loss = 0
            self.train()
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
                # if self.init_clip_max_norm is not None:
                #     torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.init_clip_max_norm)
                avg_loss += loss.item()
                eval_step = 5000 if self.args.task in ("douban", "e_commerce") else 100000
                if i % eval_step == 0 and i != 0:
                    print("evaluate ...")
                    self.evaluate(dev)
                    self.evaluate(test, is_test=True)
                    self.train()
                    # print('temp val = ', self.bert_model.temp.item())
            self.evaluate(dev)
            cnt = len(dataset) // self.args.batch_size + 1
            print("Average loss:{:.6f} ".format(avg_loss / cnt))
            avg_loss = avg_loss / cnt
            # save_dir = "./PT_checkpoint/{}_{}_epoch/".format(self.args.task, epoch)
            save_dir = "./PT_checkpoint/{}_{}_{}_epoch/".format(self.args.save_version,
                                                                self.args.task, epoch)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(self.bert_model.state_dict(), save_dir+"pytorch_model.bin")
            self.bert_model.config.save_pretrained(save_dir)
            print('save pretrain models to', save_dir)
            print('save pretrain models on epoch end ...')
            if avg_loss > before - 0.01 or epoch >= 5:
                self.adjust_learning_rate(0.8)
            before = avg_loss

    def adjust_learning_rate(self, decay_rate=.5):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
            self.args.learning_rate = param_group['lr']
        print("Decay learning rate to: ", self.args.learning_rate)


    def evaluate(self, dev, is_test=False):
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
                self.best_result[5]:
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
        collate_fn = Collate_fn(self.bert_tokenizer, mlm=False)
        dataloader = DataLoader(dataset, batch_size=50, collate_fn=collate_fn, shuffle=False)
        for i, data in enumerate(dataloader):
            with torch.no_grad():
                data = self.to_device(data)
                labels = data['labels']
                output = self.bert_model(**data)
                # logits = torch.sigmoid(output[0]).squeeze(-1)
                logits = output[0]
                y_pred += logits

            if i % 200 == 0:
                print('Batch[{}] batch_size:{}'.format(i, labels.size(0)))
        print("pred len: ", len(y_pred))
        return y_pred

    def load_model(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict=state_dict, strict=False)
        print("---load weight from ", path, "...")
        if torch.cuda.is_available(): self.cuda()


parser = argparse.ArgumentParser()
parser.add_argument("--task",
                    default='e_commerce',
                    type=str,
                    help="The dataset used for training and test.")

parser.add_argument("--model_class",
                    default='du_bert_pretrain',
                    type=str,
                    help="Model Class")

parser.add_argument("--save_version",
                    default='4',
                    type=str)

parser.add_argument("--is_training",
                    action='store_true',
                    help="Training model or testing model?")

parser.add_argument("--batch_size",
                    default=24,
                    type=int,
                    help="The batch size.")

parser.add_argument("--learning_rate",
                    default=1e-6,
                    type=float,
                    help="The initial learning rate for Adamw.")

parser.add_argument("--epochs",
                    default=15,
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
parser.add_argument("--load_fp", action='store_true')
parser.add_argument("--cl_layer", default=6, type=int)

args = parser.parse_args()
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

args.save_path += args.task + '.pretrain.' + "{}.pt".format(args.save_version)
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
        test_model(test)

    end = time.time()
    print("use time: ", (end - start) / 60, " min")