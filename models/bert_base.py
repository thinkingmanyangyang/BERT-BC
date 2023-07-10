from transformers import BertPreTrainedModel, BertModel
from torch import nn

class BertBase(BertModel):
    def __init__(self, config):
        super(BertBase, self).__init__(config)
        self.embeddings.character_embeddings = nn.Embedding(3, config.hidden_size)

if __name__ == '__main__':
    model = BertBase.from_pretrained('../pretrain_models/bert-base-uncased')
    print(model)