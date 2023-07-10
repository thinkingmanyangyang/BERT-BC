from transformers import BertTokenizer
import torch
import random

class Tokenizer(object):
    def __init__(self, pretrain_path):
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        print("bert tokenizer ...")
        # special_tokens_dict = {'eos_token': '[eos]'}
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

    def batch_encode_plus(self, contexts, responses, max_length=256, return_tensor='pt', truncation=True, type=None):
        all_input_ids = []
        all_token_type_ids = []
        all_visible_mask = []
        all_character_ids = []
        max_seq_len = 0
        for context, response in zip(contexts, responses):
            input_ids, token_type_ids, visible_mask, character_ids = \
                self.encode(context, response, max_length=max_length, truncation=truncation)
            max_seq_len = max(max_seq_len, len(input_ids))
            all_input_ids.append(input_ids)
            all_token_type_ids.append(token_type_ids)
            all_visible_mask.append(visible_mask)
            all_character_ids.append(character_ids)

        padded_input_ids = []
        padded_token_type_ids = []
        padded_attention_mask = []
        padded_visible_mask = []
        padded_character_ids = []

        for i in range(len(all_input_ids)):
            input_ids = all_input_ids[i]
            token_type_ids = all_token_type_ids[i]
            visible_mask = all_visible_mask[i]
            character_ids = all_character_ids[i]

            pad_len = max_seq_len - len(input_ids)
            attention_mask = [1] * len(input_ids) + [0] * pad_len
            input_ids = input_ids + [0] * pad_len
            token_type_ids = token_type_ids + [0] * pad_len
            visible_mask = visible_mask + [0] * pad_len
            # visible_mask = visible_mask + [visible_mask[-1]+1] * pad_len
            # visible_mask = self._padding(visible_mask, max_length=max_seq_len, pad_token=visible_mask[-1]+1)
            character_ids = character_ids + [0] * pad_len

            padded_input_ids.append(input_ids)
            padded_token_type_ids.append(token_type_ids)
            padded_attention_mask.append(attention_mask)
            padded_visible_mask.append(visible_mask)
            padded_character_ids.append(character_ids)

        raw_dict = {'input_ids': padded_input_ids,
                   'token_type_ids': padded_token_type_ids,
                   'attention_mask': padded_attention_mask,
                   'visible_mask': padded_visible_mask,
                   'character_ids': padded_character_ids
                   }
        if return_tensor == 'pt':
            for key in raw_dict:
                raw_dict[key] = torch.LongTensor(raw_dict[key])
        return raw_dict

    def batch_encode_mlm(self, contexts, responses, max_length=256, return_tensor='pt', truncation=True):
        all_input_ids = []
        all_token_type_ids = []
        all_attention_mask = []
        all_mlm_labels = []
        max_seq_len = 0
        for context, response in zip(contexts, responses):
            input_ids, token_type_ids, mlm_labels = \
                self.encode_mlm(context, response, max_length=max_length, truncation=truncation)
            max_seq_len = max(max_seq_len, len(input_ids))
            all_input_ids.append(input_ids)
            all_token_type_ids.append(token_type_ids)
            all_attention_mask.append([1] * len(input_ids))
            all_mlm_labels.append(mlm_labels)

        padded_input_ids = self._padding(all_input_ids, max_seq_len, 0)
        padded_token_type_ids = self._padding(all_token_type_ids, max_seq_len, 0)
        padded_attention_mask = self._padding(all_attention_mask, max_seq_len, 0)
        padded_mlm_labels = self._padding(all_mlm_labels, max_seq_len, -100)

        raw_dict = {'input_ids': padded_input_ids,
                    'token_type_ids': padded_token_type_ids,
                    'attention_mask': padded_attention_mask,
                    'mlm_labels': padded_mlm_labels,
                    }
        if return_tensor == 'pt':
            for key in raw_dict:
                raw_dict[key] = torch.LongTensor(raw_dict[key])
        return raw_dict


    def encode(self, context, response, max_length=250, max_utt_len=50, truncation=True):
        context_tokens = []
        utt_id = 1
        cha_id = 0
        visible_mask = []
        character_ids = []

        for u in context:
            utt_tokens = u.split(' ')
            # if len(utt_tokens) > max_utt_len:
            #     utt_tokens = utt_tokens[-max_utt_len:] + [self.tokenizer.eos_token]
            utt_tokens = utt_tokens + [self.tokenizer.eos_token]
            context_tokens += utt_tokens
            visible_mask += [utt_id] * len(utt_tokens)
            character_ids += [cha_id] * len(utt_tokens)
            utt_id += 1
            cha_id = 1-cha_id

        response_tokens = response.split(' ')
        if truncation:
            self._truncate_seq_pair(tokens_a=context_tokens, tokens_b=response_tokens, max_length=max_length)

        context_token_ids = self.tokenizer.convert_tokens_to_ids(context_tokens)
        response_token_ids = self.tokenizer.convert_tokens_to_ids(response_tokens)

        visible_mask = visible_mask[-len(context_tokens):]
        character_ids = character_ids[-len(context_tokens):]
        conv_len = visible_mask[-1] - visible_mask[0] + 1
        # conv_len = utt_id
        input_ids = [self.tokenizer.cls_token_id] + context_token_ids + [self.tokenizer.sep_token_id] + \
                    response_token_ids + [self.tokenizer.sep_token_id]
        token_type_ids = [0] + [0] * len(context_token_ids) + [0] + [1] * len(response_token_ids) + [1]

        visible_mask = [0] + visible_mask + [utt_id-1] + [utt_id]*len(response_token_ids) + [utt_id]
        # visible_mask = [utt_id+1] + visible_mask + [utt_id-1] + [utt_id]*len(response_token_ids) + [utt_id]
        character_ids = [2] + character_ids + [2] + [cha_id] * len(response_token_ids) + [2]
        # print(context_tokens, response_tokens)
        # print(input_ids, token_type_ids, visible_mask, character_ids)
        assert len(character_ids) == len(visible_mask) == len(input_ids) == len(token_type_ids)
        return input_ids, token_type_ids, visible_mask, character_ids

    def encode_mlm(self, context, response, max_length=510, max_utt_len=50, truncation=True):
        context_tokens = []
        for u in context:
            utt_tokens = u.split(' ')
            utt_tokens = utt_tokens + [self.tokenizer.eos_token]
            context_tokens += utt_tokens

        response_tokens = response.split(' ')
        if truncation:
            self._truncate_seq_pair(tokens_a=context_tokens, tokens_b=response_tokens, max_length=max_length)
        context_mlm_labels = self.random_word(context_tokens)
        response_mlm_labels = self.random_word(response_tokens)
        context_token_ids = self.tokenizer.convert_tokens_to_ids(context_tokens)
        response_token_ids = self.tokenizer.convert_tokens_to_ids(response_tokens)

        # conv_len = utt_id
        input_ids = [self.tokenizer.cls_token_id] + context_token_ids + [self.tokenizer.sep_token_id] + \
                    response_token_ids + [self.tokenizer.sep_token_id]
        token_type_ids = [0] + [0] * len(context_token_ids) + [0] + [1] * len(response_token_ids) + [1]
        mlm_labels = [-100] + context_mlm_labels + [-100] + response_mlm_labels + [-100]
        assert len(input_ids) == len(token_type_ids) == len(mlm_labels)
        return input_ids, token_type_ids, mlm_labels

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

    def tokenize(self, text, **kwargs):
        return self.tokenizer.tokenize(text, **kwargs)




