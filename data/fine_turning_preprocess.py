import json

from models.tokenizer import Tokenizer
from tqdm import tqdm
FT_model = {
    'ubuntu': 'bert-base-uncased',
    'douban': 'bert-base-chinese',
    'e_commerce': 'bert-base-chinese'
}

FT_data={
    'ubuntu': 'data/ubuntu_data',
    'douban': 'data/douban_data',
    'e_commerce': 'data/e_commerce_data'
}

data_type = "ubuntu"
print(data_type)
tokenizer = Tokenizer(FT_model[data_type])


def save_json(data, file_name):
    with open(file_name, 'w', encoding='utf8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def preprocess(file_name, is_train=False):
    with open(file_name, encoding='utf8') as f:
        data = []
        for line in tqdm(f):
            line = line.strip()
            lines = line.split('\t')
            label, context, response = int(lines[0]), lines[1:-1], lines[-1]
            data.append([label, context, response])
    print('data loaded...')
    # 删除douban 最后一条数据
    if len(data) == 50001:
        data = data[:50000]

    preprocessed_data = []
    def add_bos_eos(u):
        return "[bos]" + u + "[eos]"
    for label, context, response in tqdm(data):
        assert len(response) != 0
        context = [" ".join(tokenizer.tokenize(u)) for u in context]
        response = " ".join(tokenizer.tokenize(response))
        if is_train:
            new_context = []
            for c in context:
                if len(c) != 0:
                    new_context.append(c)
            context = new_context
            if len(context) == 0:
                continue
            while len(response) == 0 and len(context) > 1:
                context = context[:-1]
                response = context[-1]
            if len(response) == 0:
                continue
            for c in context:
                assert len(c) != 0
            assert len(response) != 0
        preprocessed_data.append({
            'context': context,
            'response': response,
            'label': label
        })
    return preprocessed_data

file_dir = FT_data[data_type]

dev = preprocess(file_dir + "/dev.txt", is_train=False)
test = preprocess(file_dir + "/test.txt", is_train=False)
train = preprocess(file_dir + "/train.txt", is_train=True)


data = {"train": train, "dev": dev, "test": test}
save_json(data, file_dir+"/tokenized_data.json")

# with open('data/douban_data/tokenized_data.json', encoding='utf8') as f:
#     data = json.load(f)
#
# train = data['train']
# dev = data['dev']
# test = data['test']
# print(len(train), len(dev), len(test))