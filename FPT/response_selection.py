import time
import argparse
import pickle
import os
import json
# from setproctitle import setproctitle
from bert_fineturning import NeuralNetwork

# setproctitle('BERT_FP')
print(os.getcwd())
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#Dataset path.
FT_data = {
    'ubuntu': '../data/ubuntu_data/tokenized_data.json',
    'douban': '../data/douban_data/tokenized_data.json',
    'e_commerce': '../data/e_commerce_data/tokenized_data.json'
}
print(os.getcwd())
## Required parameters
parser = argparse.ArgumentParser()
parser.add_argument("--task",
                    default='e_commerce',
                    type=str,
                    help="The dataset used for training and test.")

parser.add_argument("--model_class",
                    default='bert',
                    type=str,
                    help="Model Class")

parser.add_argument("--save_version",
                    default='0',
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
                    default=5,
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


if __name__ == '__main__':
    start = time.time()
    data = load_json(FT_data[args.task])
    train, dev, test = data['train'], data['dev'], data['test']
    # with open(FT_data[args.task], 'rb') as f:
    #     train, dev, test = pickle.load(f, encoding='ISO-8859-1')
    print('data loaded ... ...')
    if args.is_training==True:
        # train_model(train, dev)
        model = NeuralNetwork(args=args)
        model.fit(train, dev, test)
        test_model(test)
    else:
        test_model(test)

    end = time.time()
    print("use time: ", (end - start) / 60, " min")




