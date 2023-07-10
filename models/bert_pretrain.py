import random
import torch
import transformers
from transformers import BertPreTrainedModel, BertModel
if transformers.__version__ <= "3.4.0":
    from transformers.modeling_bert import BertPreTrainingHeads
else:
    from transformers.models.bert.modeling_bert import BertPreTrainingHeads, BertForPreTraining
from torch.nn import CrossEntropyLoss


class BertPretrain(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                mlm_labels=None,
                labels=None,
                ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if mlm_labels is not None and labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 3), labels.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        probs = torch.softmax(seq_relationship_score, dim=-1)[:, 2].contiguous()
        outputs = (probs.tolist(), total_loss)
        return outputs

