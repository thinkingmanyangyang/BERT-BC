from transformers import BertTokenizer
import random
import torch

class Tokenizer(object):
    def __init__(self, pretrain_path):
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        print("du bert tokenizer....")
        # special_tokens_dict = {'eos_token': '[eos]', 'bos_token': '[bos]'}
        # 用bos 代表begin of conversation, eos 分割conversation 中的 句子
        special_tokens_dict = {'eos_token': '[eos]', 'bos_token': '[bos]'}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

    def __len__(self):
        return len(self.tokenizer)

    def _padding(self, input_list, max_length, pad_token=0):
        padded_result = []
        for li in input_list:
            pad_len = max_length - len(li)
            padded_result.append(li + [pad_token] * pad_len)
        return padded_result

    def batch_encode_plus(self, contexts, responses, return_tensor='pt', truncation=True, type='fine-turning',
                          max_context_len=190, max_response_len=70):
        '''
        :param contexts:
        :param responses:
        :param return_tensor:
        :param truncation:
        :param type: ['fine-turning', 'pretrain']
        :param max_context_len:
        :param max_response_len:
        :return:
        '''
        all_context_input_ids = []
        all_response_input_ids = []
        all_context_token_type_ids = []
        all_response_token_type_ids = []
        all_context_attention_mask = []
        all_response_attention_mask = []
        all_context_character_ids = []
        all_response_character_ids = []
        if type == 'pretrain':
            all_context_mlm_labels = []
            all_response_mlm_labels = []

        max_ctx_len = 0
        max_res_len = 0
        for context, response in zip(contexts, responses):
            if type == 'fine-turning':
                context_input_ids, context_token_type_ids, context_character_ids,\
                    response_input_ids, response_token_type_ids, response_character_ids = \
                    self.encode(context, response, truncation=truncation,
                                max_context_len=max_context_len-1, max_response_len=max_response_len-1)
            # pretrain 也需要修改一下，8.29 暂时没有修改
            elif type == 'pretrain':
                context_input_ids, context_token_type_ids, context_character_ids, context_mlm_labels, \
                response_input_ids, response_token_type_ids, response_character_ids, response_mlm_labels = \
                    self.encode_with_mlm(context, response, truncation=truncation,
                                         max_context_len=max_context_len-1, max_response_len=max_response_len-1)
                all_context_mlm_labels.append(context_mlm_labels)
                all_response_mlm_labels.append(response_mlm_labels)

            max_ctx_len = max(max_ctx_len, len(context_input_ids))
            max_res_len = max(max_res_len, len(response_input_ids))
            all_context_input_ids.append(context_input_ids)
            all_response_input_ids.append(response_input_ids)
            all_context_attention_mask.append([1] * len(context_input_ids))
            all_response_attention_mask.append([1] * len(response_input_ids))
            all_context_token_type_ids.append(context_token_type_ids)
            all_response_token_type_ids.append(response_token_type_ids)
            all_context_character_ids.append(context_character_ids)
            all_response_character_ids.append(response_character_ids)

        max_ctx_len = min(max_context_len, max_ctx_len + 1)
        max_res_len = min(max_response_len, max_res_len + 1)
        padded_context_input_ids = self._padding(all_context_input_ids, max_ctx_len)
        padded_response_input_ids = self._padding(all_response_input_ids, max_res_len)
        padded_context_token_type_ids = self._padding(all_context_token_type_ids, max_ctx_len)
        padded_response_token_type_ids = self._padding(all_response_token_type_ids, max_res_len)
        padded_context_attention_mask = self._padding(all_context_attention_mask, max_ctx_len)
        padded_response_attention_mask = self._padding(all_response_attention_mask, max_res_len)
        padded_context_character_ids = self._padding(all_context_character_ids, max_ctx_len)
        padded_response_character_ids = self._padding(all_response_character_ids, max_res_len)

        raw_dict = {'context_input_ids': padded_context_input_ids,
                   'context_token_type_ids': padded_context_token_type_ids,
                   'context_attention_mask': padded_context_attention_mask,
                   'context_character_ids': padded_context_character_ids,
                   'response_input_ids': padded_response_input_ids,
                   'response_token_type_ids': padded_response_token_type_ids,
                   'response_attention_mask': padded_response_attention_mask,
                   'response_character_ids': padded_response_character_ids,
                   }
        if type == 'pretrain':
            padded_context_mlm_labels = self._padding(all_context_mlm_labels, max_ctx_len, -100)
            padded_response_mlm_labels = self._padding(all_response_mlm_labels, max_res_len, -100)
            raw_dict['context_mlm_labels'] = padded_context_mlm_labels
            raw_dict['response_mlm_labels'] = padded_response_mlm_labels

        if return_tensor == 'pt':
            for key in raw_dict:
                raw_dict[key] = torch.LongTensor(raw_dict[key])
        return raw_dict

    def encode(self, context, response, max_context_len=190, max_utt_len=50, max_response_len=70, truncation=True):
        context_tokens = []
        cha_id = 0
        character_ids = []
        for u in context:
            utt_tokens = u.split(' ')
            # if len(utt_tokens) > max_utt_len:
            #     utt_tokens = utt_tokens[-max_utt_len:]
            utt_tokens = utt_tokens + [self.tokenizer.eos_token]
            context_tokens += utt_tokens
            character_ids += [cha_id] * len(utt_tokens)
            cha_id = 1-cha_id

        response_tokens = response.split(' ')
        response_tokens = [self.tokenizer.bos_token] + response_tokens

        # if truncation:
        #     self._truncate_seq_pair(tokens_a=context_tokens, tokens_b=response_tokens, max_length=max_length)
        # -2 后面还有 eos sep
        response_tokens = response_tokens[: max_response_len-2]
        # -3 有 cls bos sep
        context_tokens = context_tokens[-max_context_len+3:]
        character_ids = character_ids[-len(context_tokens):]

        context_tokens = [self.tokenizer.cls_token, self.tokenizer.bos_token] + context_tokens + [self.tokenizer.sep_token]
        response_tokens = response_tokens + [self.tokenizer.eos_token, self.tokenizer.sep_token]
        character_ids = [character_ids[0]] * 2 + character_ids + [character_ids[-1]]
        # print(context_tokens, response_tokens)
        context_token_ids = self.tokenizer.convert_tokens_to_ids(context_tokens)
        response_token_ids = self.tokenizer.convert_tokens_to_ids(response_tokens)

        context_character_ids = character_ids
        response_character_ids = [cha_id] * len(response_token_ids)

        context_input_ids = context_token_ids
        response_input_ids = response_token_ids

        context_token_type_ids = [0] * len(context_token_ids)
        response_token_type_ids = [1] * len(response_token_ids)
        return context_input_ids, context_token_type_ids, context_character_ids, \
               response_input_ids, response_token_type_ids, response_character_ids


    def encode_with_mlm(self, context, response, max_context_len=190, max_utt_len=50, max_response_len=70, truncation=True):
        '''
        :param context:
        :param response:
        :param max_context_len:
        :param max_utt_len:
        :param max_response_len:
        :param truncation:
        :return:
                context_token_ids = [cls, bos, u1, eos, u2, eos, u3, eos, sep]
                response_token_ids = [cls, bos, un, eos, sep]
        '''
        context_tokens = []
        context_mlm_labels = []
        cha_id = 0
        character_ids = []
        for u in context:
            utt_tokens = u.split(' ')
            # if len(utt_tokens) > max_utt_len:
            #     utt_tokens = utt_tokens[-max_utt_len:]
            # mlm_labels = self.random_word(utt_tokens) + [-100]
            mlm_labels = self.random_word(utt_tokens) + [self.tokenizer.eos_token_id]
            utt_tokens = utt_tokens + [self.tokenizer.eos_token]
            context_tokens += utt_tokens
            context_mlm_labels += mlm_labels
            character_ids += [cha_id] * len(utt_tokens)
            cha_id = 1 - cha_id
        # 这里没有像上面那样添加 bos token 是避免后面生成mask token的时候 bos 被mask掉
        response_tokens = response.split(' ')
        response_tokens = response_tokens[:max_response_len-3]
        # 截断
        response_mlm_labels = self.random_word(response_tokens)
        # 截断
        context_tokens = context_tokens[-max_context_len+3:]
        context_mlm_labels = context_mlm_labels[-max_context_len+3:]
        character_ids = character_ids[-max_context_len+3:]

        context_tokens = [self.tokenizer.cls_token, self.tokenizer.bos_token] + context_tokens + [
            self.tokenizer.sep_token]
        context_mlm_labels = [-100] * 2 + context_mlm_labels + [-100]
        # cls 处的character ids 用于最终和response 去分类， 所以取 response 对应的character id
        character_ids = [cha_id, character_ids[0]] + character_ids + [character_ids[-1]]

        response_tokens = [self.tokenizer.bos_token] + response_tokens + [self.tokenizer.eos_token, self.tokenizer.sep_token]
        # response_mlm_labels = [-100] + response_mlm_labels + [-100] * 2
        response_mlm_labels = [-100] + response_mlm_labels + [self.tokenizer.eos_token_id, -100]

        context_token_ids = self.tokenizer.convert_tokens_to_ids(context_tokens)
        response_token_ids = self.tokenizer.convert_tokens_to_ids(response_tokens)

        context_character_ids = character_ids
        response_character_ids = [cha_id] * len(response_token_ids)

        context_input_ids = context_token_ids
        response_input_ids = response_token_ids

        context_token_type_ids = [0] * len(context_token_ids)
        response_token_type_ids = [1] * len(response_token_ids)
        return context_input_ids, context_token_type_ids, context_character_ids, context_mlm_labels,\
                response_input_ids, response_token_type_ids, response_character_ids, response_mlm_labels


    def random_word(self, tokens):
        output_label = []
        for i, token in enumerate(tokens):
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                prob /= 0.15
                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = "[MASK]"
                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.choice(list(self.tokenizer.vocab.items()))[0]
                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)f
                try:
                    output_label.append(self.tokenizer.vocab[token])
                except KeyError:
                    # For unknown words (should not occur with BPE vocab)
                    output_label.append(self.tokenizer.vocab["[UNK]"])
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-100)
        return output_label


    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop(0)
            else:
                tokens_b.pop()

    def tokenize(self, text, **kwargs):
        return self.tokenizer.tokenize(text, **kwargs)

if __name__ == '__main__':
    tokenizer = Tokenizer(pretrain_path='../pretrain_models/bert-base-chinese')
    text = "亲 亲 真 的 不 好 意 思 我 们 已 经 是 优 惠 价 了 呢 小 本 生 意 请 亲 谅 解".split(' ')
    tokenizer.random_word(text)

    res = tokenizer.batch_encode_plus([["亲 亲 真 的 不 好 意 思 我 们 已 经 是 优 惠 价 了 呢 小 本 生 意 请 亲 谅 解"]], ["好 的"], max_context_len=25, max_response_len=3)
    print(text)
    print(res)
    context_ids = res["context_input_ids"][0]
    print(context_ids)
    print(len(context_ids))
    print(tokenizer.tokenizer.decode(res["context_input_ids"][0], spaces_between_special_tokens=False))
    print(tokenizer.tokenizer.convert_ids_to_tokens(context_ids))

