import random
import torch
from transformers import BertPreTrainedModel, BertModel
# from transformers.modeling_bert import BertPreTrainingHeads
from torch import nn
from torch.nn import CrossEntropyLoss, BCELoss
# from loss_function import LabelSmoothLoss


class CustomMaskModel(BertPreTrainedModel):
    def __init__(self, config):
        super(CustomMaskModel, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()


    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                visible_mask=None,
                character_ids=None,
                labels=None
                ):
        if visible_mask is not None:
            attention_mask = self.create_attention_mask(
                visible_mask=visible_mask,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            # print("call custom mask model")
        # print(input_ids.shape)
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        probs = torch.sigmoid(logits).squeeze()
        loss_fct = BCELoss()
        loss = loss_fct(probs, target=labels)

        outputs = (probs.tolist(),) + (loss,)
        return outputs

    def create_attention_mask(self, visible_mask, attention_mask, token_type_ids=None):
        mask = visible_mask[:, :, None] >= visible_mask[:, None, :]
        mask = mask.to(visible_mask.dtype)
        attention_mask = attention_mask[:, None, :]
        attention_mask = mask & attention_mask
        attention_mask[:, 0, :] = 1
        return attention_mask

    def create_tacm_mask(self, visible_mask, attention_mask, token_type_ids):
        # utterance 关注自己的 mask
        mask1 = visible_mask[:, :, None] == visible_mask[:, None, :]
        # response 关注的 mask
        mask2 = token_type_ids[:, :, None] | token_type_ids[:, None, :]
        mask1 = mask1.long()
        mask = mask1 | mask2

        mask = mask & attention_mask[:, None, :]
        mask = mask.long()
        # cls 关注的 mask
        mask[:, 0, :] = 1
        mask[:, :, 0] = 1
        # print(mask.long())
        return mask

    def random_choice(self, input_ids, token_type_ids, attention_mask, type=1):
        # type 0, no random choice
        # type 1, target sentence random choice
        # type 2, source and target all use random choice, in this type,
        #         you should use mlm task for source sentence
        if type == 0:
            return input_ids
        # random choice
        if random.random() < 0.5:
            input_ids = self._random_choice(input_ids, target_mask=token_type_ids)
        elif type == 2:
            input_ids = self._random_choice(
                input_ids=input_ids,
                target_mask=(1 - token_type_ids) & attention_mask
            )
        return input_ids

    def _random_choice(self, input_ids, target_mask):
        batch_size, sequence_length = input_ids.shape
        device = input_ids.device
        vocab_size = self.bert.embeddings.word_embeddings.weight.shape[0]
        rand_ids = torch.rand(batch_size, sequence_length) * vocab_size
        rand_ids = rand_ids.to(input_ids.dtype).to(device) % vocab_size
        rand_mask = torch.rand(batch_size, sequence_length).to(device) > 0.3
        disturb_ids = torch.where(rand_mask, input_ids, rand_ids)
        input_ids = torch.where(target_mask == 0, input_ids, disturb_ids)
        return input_ids

    def compute_loss(self, predictions, labels, target_mask):
        # compute loss for batch
        vocab_size = predictions.shape[-1]
        predictions = predictions.view(-1, vocab_size)
        labels = labels.view(-1)
        target_mask = target_mask.view(-1).float()
        loss = self.loss_fct(predictions, labels) * target_mask
        return loss.sum() / target_mask.sum()

    def compute_loss_for_sentence(self, predictions, labels, target_mask):
        # compute loss for sentence, then compute the mean of loss
        batch_size = predictions.shape[0]
        loss = 0.0
        for i in range(batch_size):
            loss += self.compute_loss(predictions[i], labels[i], target_mask[i])
        return loss / batch_size









