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

with open('ubuntu_data/ubuntu_post_train.json', encoding='utf8') as f:
    data = json.load(f)

tokenized_data = []
cnt = 0
for conv in tqdm(data):
    tokenized_conv = []
    flag = 0
    for sent in conv:
        sent = tokenizer.tokenizer.tokenize(sent)
        sent = " ".join(sent)
        if len(sent.strip()) == 0:
            cnt += 1
            continue
        assert len(sent.split(' ')) != 0
        tokenized_conv.append(sent)
    tokenized_data.append(tokenized_conv)
print(cnt)

with open('ubuntu_data/ubuntu_tokenized_post_train.json', 'w', encoding='utf8') as f:
    json.dump(tokenized_data, f, indent=4, ensure_ascii=False)

