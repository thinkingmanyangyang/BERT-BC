# align + resample + sim cse + wti + pretrain
import random
import torch
import transformers
from transformers import BertPreTrainedModel, BertModel, BertForPreTraining
if transformers.__version__ <= "3.4.0":
    from transformers.modeling_bert import BertPreTrainingHeads
else:
    from transformers.models.bert.modeling_bert import BertPreTrainingHeads

from torch import nn
from torch.nn import CrossEntropyLoss, BCELoss
from models.glu import GLU
import torch.nn.functional as F
# from loss_function import LabelSmoothLoss

class PolyEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(PolyEncoder, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)
        # self.classifier = nn.Linear(config.hidden_size, 2)

        contrastive_dim = 256
        temp = 0.05
        fg_temp = 0.05
        self.cl_layer = config.cl_layer
        self.queue_size = 64
        self.context_transform = nn.Linear(config.hidden_size, contrastive_dim)
        self.response_transform = nn.Linear(config.hidden_size, contrastive_dim)
        self.context_glu = GLU(config.hidden_size, 256)
        self.response_glu = GLU(config.hidden_size, 256)
        self.context_fg_transform = nn.Linear(config.hidden_size, contrastive_dim)
        self.response_fg_transform = nn.Linear(config.hidden_size, contrastive_dim)

        self.context_weight_fc = nn.Linear(contrastive_dim, 1)
        self.response_weight_fc = nn.Linear(contrastive_dim, 1)
        self.temp = nn.Parameter(torch.ones([]) * temp)
        self.fg_temp = nn.Parameter(torch.ones([]) * fg_temp)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        self.poly_code_embeddings = nn.Embedding(360, config.hidden_size)
        self.poly_context = nn.Linear(config.hidden_size, 256)
        self.poly_response = nn.Linear(config.hidden_size, 256)
        self.init_weights()

        # # create the queue
        # self.register_buffer("context_queue", torch.randn(contrastive_dim, self.queue_size))
        # self.register_buffer("response_queue", torch.randn(contrastive_dim, self.queue_size))
        # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # self.context_queue = nn.functional.normalize(self.context_queue, dim=0)
        # self.response_queue = nn.functional.normalize(self.response_queue, dim=0)

    def forward(self,
                context_input_ids=None,
                context_attention_mask=None,
                context_token_type_ids=None,
                context_character_ids=None,
                response_input_ids=None,
                response_attention_mask=None,
                response_token_type_ids=None,
                response_character_ids=None,
                labels=None,
                context_mlm_labels=None,
                response_mlm_labels=None,
                neg_threshold=None,
                ):

        device = context_input_ids.device
        if labels is not None:
            labels = labels.long()
        # extend_context_attention_mask = self.extend_attention_mask(context_attention_mask)
        # extend_response_attention_mask = self.extend_attention_mask(response_attention_mask)

        encoder_extended_attention_mask = None
        encoder_hidden_states = None

        head_mask = [None] * self.config.num_hidden_layers
        # print(context_input_ids.device)
        # print(self.bert.embeddings.word_embeddings.weight.device)
        context_embedding_output = self.bert.embeddings(
            input_ids=context_input_ids, token_type_ids=context_token_type_ids
        )
        response_embedding_output = self.bert.embeddings(
            input_ids=response_input_ids, token_type_ids=response_token_type_ids
        )
        cl_layer = self.cl_layer
        context_encoder_outputs = self.encoder_forward(
            context_embedding_output,
            attention_mask=self.extend_attention_mask(context_attention_mask),
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            start_layer=0, end_layer=12
        )
        response_encoder_outputs = self.encoder_forward(
            response_embedding_output,
            attention_mask=self.extend_attention_mask(response_attention_mask),
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            start_layer=0, end_layer=12
        )

        # bs, ctx_len, hidden
        context_output = context_encoder_outputs[0]
        # bs, res_len, hidden
        response_output = response_encoder_outputs[0]
        bs, res_len, hidden_size = response_output.shape
        # compute label loss
        loss_fct = CrossEntropyLoss()
        if labels is not None:
            # loss 1
            response_output = response_output[None, ...]
            # 将batch内所有response都看做候选回复
            response_output = response_output.expand(bs, bs, res_len, hidden_size).contiguous()
            # bs, res_cnt=bs
            sim_matrix = self.poly_sim(context_embedding=context_output, response_embedding=response_output)
            sim_matrix = sim_matrix / self.temp
            poly_loss = self.compute_contrastive_loss(sim_c2r=sim_matrix, sim_r2c=sim_matrix.t(), labels=labels)
            # next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), labels.view(-1))
            total_loss = poly_loss
            probs = None
        else:
            total_loss = 0
            response_output = response_output[:, None, ...].contiguous()
            # bs, 1
            sim_matrix = self.poly_sim(context_embedding=context_output, response_embedding=response_output)
            probs = sim_matrix.reshape(bs).tolist()
        # probs = torch.softmax(seq_relationship_score, dim=-1)[:, 1].contiguous()
        outputs = (probs,)

        # self._dequeue_and_enqueue(true_context.clone(), true_response.clone())
        outputs = outputs + (total_loss,)
        return outputs

    def get_context_embedding(self,
                              context_input_ids=None,
                              context_attention_mask=None,
                              context_token_type_ids=None,
                              **kwargs
                              ):
        device = context_input_ids.device
        # extend_context_attention_mask = self.extend_attention_mask(context_attention_mask)
        # extend_response_attention_mask = self.extend_attention_mask(response_attention_mask)

        encoder_extended_attention_mask = None
        encoder_hidden_states = None

        head_mask = [None] * self.config.num_hidden_layers
        # print(context_input_ids.device)
        # print(self.bert.embeddings.word_embeddings.weight.device)
        context_embedding_output = self.bert.embeddings(
            input_ids=context_input_ids, token_type_ids=context_token_type_ids
        )
        cl_layer = self.cl_layer
        context_encoder_outputs = self.encoder_forward(
            context_embedding_output,
            attention_mask=self.extend_attention_mask(context_attention_mask),
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            start_layer=0, end_layer=cl_layer
        )
        context_output = context_encoder_outputs[0]
        context_pooled_output = context_output[:, 1, :]
        context_feat = F.normalize(self.context_transform(context_pooled_output), dim=-1)
        return context_output

    def get_response_embedding(self,
                               response_input_ids=None,
                               response_attention_mask=None,
                               response_token_type_ids=None,
                               **kwargs
                               ):
        device = response_input_ids.device
        # extend_context_attention_mask = self.extend_attention_mask(context_attention_mask)
        # extend_response_attention_mask = self.extend_attention_mask(response_attention_mask)

        encoder_extended_attention_mask = None
        encoder_hidden_states = None

        head_mask = [None] * self.config.num_hidden_layers
        # print(context_input_ids.device)
        # print(self.bert.embeddings.word_embeddings.weight.device)
        response_embedding_output = self.bert.embeddings(
            input_ids=response_input_ids, token_type_ids=response_token_type_ids
        )
        cl_layer = self.cl_layer
        response_encoder_outputs = self.encoder_forward(
            response_embedding_output,
            attention_mask=self.extend_attention_mask(response_attention_mask),
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            start_layer=0, end_layer=cl_layer
        )
        response_output = response_encoder_outputs[0]
        response_pooled_output = response_output[:, 0, :]
        response_feat = F.normalize(self.response_transform(response_pooled_output), dim=-1)
        return response_output

    def cross_encoder_forward(self, context_hidden, response_hidden):
        concat_hidden = torch.cat([context_hidden, response_hidden], dim=1)
        # bs, seq_len
        # concat_attention_mask = torch.cat([context_attention_mask, response_attention_mask], dim=-1)
        concat_outputs = self.encoder_forward(
            concat_hidden,
            attention_mask=None,
            head_mask=[None]*12,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            start_layer=self.cl_layer, end_layer=12
        )
        return concat_outputs

    def poly_attention(self, query, key, value):
        '''
        :param query: bs, seq_len, hidden
        :param key: bs, seq_len, hidden
        :param value: bs, seq_len, hidden
        :return:
        '''
        attn_weights = torch.einsum("bqh,bkh->bqk", [query, key])
        attn_weights = torch.softmax(attn_weights, dim=-1)
        output = torch.einsum("bqk,bkh->bqh", [attn_weights, value])
        return output

    def poly_sim(self, context_embedding, response_embedding, poly_m=64):
        '''
        :param context_embedding: batch size, seq_len, hidden_size
        :param response_embedding: batch size, res cnt, seq_len, hidden_size
        :param poly_m:
        :return:
        '''
        batch_size, res_cnt, _, hidden_size = response_embedding.shape
        poly_code_ids = torch.arange(poly_m, dtype=torch.long, device=context_embedding.device)
        poly_code_ids += 1
        poly_code_ids = poly_code_ids.unsqueeze(0).expand(batch_size, poly_m)
        poly_codes = self.poly_code_embeddings(poly_code_ids)
        context_vecs = self.poly_attention(poly_codes, context_embedding, context_embedding)
        poly_code_ids = torch.zeros(batch_size * res_cnt, 1, dtype=torch.long, device=context_embedding.device)
        poly_codes = self.poly_code_embeddings(poly_code_ids)
        responses_vec = self.poly_attention(poly_codes, response_embedding.reshape(batch_size*res_cnt, -1, hidden_size),
                                            response_embedding.reshape(batch_size*res_cnt, -1, hidden_size))
        responses_vec = responses_vec.view(batch_size, res_cnt, -1)
        # # # batch, poly_n, hidden
        # context_vecs = self.poly_context(context_vecs)
        # # # batch, res_cnt, hidden
        # responses_vec = self.poly_response(responses_vec)
        # batch, res_cnt, hidden
        final_context_vec = self.poly_attention(responses_vec, context_vecs, context_vecs)
        final_context_vec = self.poly_context(final_context_vec)
        responses_vec = self.poly_response(responses_vec)
        final_context_vec = F.normalize(final_context_vec, 2, -1)
        responses_vec = F.normalize(responses_vec, 2, -1)
        # bs, res_cnt
        dot_product = torch.sum(final_context_vec * responses_vec, -1)
        return dot_product

    @torch.no_grad()
    def _dequeue_and_enqueue(self, context_feat, response_feat):
        # gather keys before updating queue
        batch_size = context_feat.shape[0]
        ptr = int(self.queue_ptr)
        batch_size = min(batch_size, self.context_queue.shape[1] - ptr)
        context_feat = context_feat[:batch_size]
        response_feat = response_feat[:batch_size]
        # assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.context_queue[:, ptr:ptr + batch_size] = context_feat.T
        self.response_queue[:, ptr:ptr + batch_size] = response_feat.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr

    def get_sim_targets(self, sim, labels):
        '''
        :param sim: batch size, extend feat num(batch + momentum): softmax(dim=1) 得到该样本与所有对比feats的相似度
        :param labels: batch size
        :return:
        '''
        sim_targets = torch.zeros(sim.size()).to(labels.device)
        sim_targets.fill_diagonal_(1)
        sim_targets = torch.masked_fill(sim_targets, labels[:, None] == 0, 0)
        return sim_targets.detach()

    def extend_attention_mask(self, attention_mask):
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def encoder_forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            start_layer=0,
            end_layer=12,
    ):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.bert.encoder.layer):
            if i < start_layer: continue
            if i >= end_layer: continue

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

    def create_attention_mask(self, visible_mask, attention_mask):
        mask = visible_mask[:, :, None] >= visible_mask[:, None, :]
        mask = mask.to(visible_mask.dtype)
        attention_mask = attention_mask[:, None, :]
        return mask & attention_mask

    def fine_grained_interaction(self, text_feat, video_feat, text_mask, video_mask, wti=False):
        # from https://github.com/foolwood/DRL
        '''
        :param text_feat: bs, seq_len1, hidden: context
        :param video_feat: bs, seq_len2, hidden: context
        :param text_mask: bs, seq_len1
        :param video_mask: bs, seq_len2
        :return:
        '''
        # compute weight
        text_weight = self.context_weight_fc(text_feat).squeeze(2)  # B x N_t x D -> B x N_t
        text_weight.masked_fill_(torch.as_tensor((1 - text_mask), dtype=torch.bool), -10000.0)
        text_weight = torch.softmax(text_weight, dim=-1)  # B x N_t

        video_weight = self.response_weight_fc(video_feat).squeeze(2)  # B x N_v x D -> B x N_v
        video_weight.masked_fill_(torch.as_tensor((1 - video_mask), dtype=torch.bool), -10000.0)
        video_weight = torch.softmax(video_weight, dim=-1)  # B x N_v

        text_feat = F.normalize(text_feat, p=2, dim=2)
        video_feat = F.normalize(video_feat, p=2, dim=2)

        retrieve_logits = torch.einsum('atd,bvd->abtv', [text_feat, video_feat])
        retrieve_logits = torch.einsum('abtv,at->abtv', [retrieve_logits, text_mask])
        retrieve_logits = torch.einsum('abtv,bv->abtv', [retrieve_logits, video_mask])

        # t2v, v2t
        t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
        v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv

        if wti:
            t2v_logits = torch.einsum('abt,at->ab', [t2v_logits, text_weight])
            v2t_logits = torch.einsum('abv,bv->ab', [v2t_logits, video_weight])
        else:
            text_sum = text_mask.sum(-1)
            video_sum = video_mask.sum(-1)
            t2v_logits = torch.sum(t2v_logits, dim=2) / (text_sum.unsqueeze(1))
            v2t_logits = torch.sum(v2t_logits, dim=2) / (video_sum.unsqueeze(0))
        retrieve_logits = (t2v_logits + v2t_logits) / 2.0
        retrieve_logits = retrieve_logits / self.fg_temp
        return retrieve_logits, retrieve_logits.T

    def global_to_fine_grained(self, cg_feats, cl_feats, cl_mask, rg_feats, rl_feats, rl_mask):
        # compute weight
        cl_weight = self.context_weight_fc(cl_feats).squeeze(2)  # B x N_t x D -> B x N_t
        # text_weight.masked_fill_(torch.as_tensor((1 - text_mask), dtype=torch.bool), float("-inf"))
        cl_weight.masked_fill_(torch.as_tensor((1 - cl_mask), dtype=torch.bool), -10000.0)
        cl_weight = torch.softmax(cl_weight, dim=-1)  # B x N_t

        rl_weight = self.response_weight_fc(rl_feats).squeeze(2)  # B x N_v x D -> B x N_v
        rl_weight.masked_fill_(torch.as_tensor((1 - rl_mask), dtype=torch.bool), -10000.0)
        rl_weight = torch.softmax(rl_weight, dim=-1)  # B x N_v

        cl_feats = F.normalize(cl_feats, p=2, dim=2)
        rl_feats = F.normalize(rl_feats, p=2, dim=2)
        # cl_feats: bs, context_len, hidden_size
        # rl_feats: bs, response_len, hidden_size
        c2r_logits = torch.einsum("ad,bvd->abv", [cg_feats, rl_feats])
        r2c_logits = torch.einsum("bd,atd->abt", [rg_feats, cl_feats])

        c2r_logits = torch.einsum("abv,bv->ab", [c2r_logits, rl_weight])
        r2c_logits = torch.einsum("abt,at->ab", [r2c_logits, cl_weight])
        retrieve_logits = (c2r_logits + r2c_logits) / 2.0
        retrieve_logits = retrieve_logits / self.fg_temp
        return retrieve_logits, retrieve_logits.T

    def inner_g2l(self):
        return

    def sim_cse_loss(self, sim_matrix, temp=0.05):
        batch_size = sim_matrix.shape[0]
        sim_matrix = sim_matrix - torch.eye(batch_size).to(sim_matrix.device) * 1e12
        sim_matrix = sim_matrix / temp

        y_true = torch.cat([torch.arange(1, batch_size, step=2, dtype=torch.long).unsqueeze(1),
                            torch.arange(0, batch_size, step=2, dtype=torch.long).unsqueeze(1)],
                           dim=1).reshape([batch_size, ]).to(sim_matrix.device)
        # sim_matrix = sim_matrix
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(sim_matrix, y_true)
        return loss

    def compute_contrastive_loss(self, sim_c2r, sim_r2c, labels, dusoftmax=False):
        # c2r
        #     r1+ r1- r2+ r2-
        # c1
        # c1
        # c2
        # c2
        # r2c
        #     c1 c1 c2 c2
        # r1+
        # r1-
        # r2+
        # r2-
        bs = sim_c2r.shape[0]
        c2r_targets = self.get_sim_targets(sim_c2r, labels)
        r2c_targets = self.get_sim_targets(sim_r2c, labels)
        # r2c_targets = c2r_targets.t()
        # sim_r2c = sim_c2r.t()

        loss_c2r = -torch.sum(F.log_softmax(sim_c2r[::2], dim=1) * c2r_targets[::2], dim=1).mean()
        # [c1, c1, c2, c2, c3, c3, ...] 的对比分数, 避免对比两遍 context

        sim_r2c = sim_r2c[:, ::2]
        r2c_targets = r2c_targets[:, ::2]
        #     c1 c2 c3 c4
        # r1+
        # r1-
        # r2+
        # r2-
        if dusoftmax:
            du_temp = 100
            sim_r2c = sim_r2c * F.softmax(sim_r2c / du_temp, dim=0) * len(sim_r2c)
        loss_r2c = -torch.sum(F.log_softmax(sim_r2c, dim=1) * r2c_targets, dim=1).mean()
        contrastive_loss = (loss_c2r + loss_r2c) / 2
        return contrastive_loss

    def hidden_mixup(self, hidden1, hidden2, l=0.15):
        mixed_hidden = l * hidden1 + (1 - l) * hidden2
        return mixed_hidden

    def token_mixup(self, hidden1, hidden2, l=0.15):
        '''
        :param hidden1: bs, seq_len, hidden
        :param hidden2: bs, seq_len, hidden
        :param l: 0-1
        :return:
        '''
        device = hidden1.device
        if len(hidden1.shape) == 2:
            seq_len, hidden_size = hidden1.shape
            rand_mask = torch.rand(seq_len, 1).to(device) > l
            mix_hidden = torch.where(rand_mask, hidden1, hidden2)
        else:
            bs, seq_len, hidden_size = hidden1.shape
            rand_mask = torch.rand(bs, seq_len, 1).to(device) > l
            mix_hidden = torch.where(rand_mask, hidden1, hidden2)
        return mix_hidden