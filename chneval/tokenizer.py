#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tokenization with Vocabulary Involved
@author: ZhiruoWang
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import random
import logger
import collections
import unicodedata


# Constants
# special token ids
PAD_ID = 0
UNK_ID = 100
CLS_ID = 101
SEP_ID = 102
MASK_ID = 103

# special token words
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"


# basic functions
def whitespace_tokenize(text):
    text = text.strip()
    if not text:
        return []
    return text.split()


def _is_whitespace(char):
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


def _is_punctuation(char):
    cp = ord(char)
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
        (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: {}".format(type(text)))


def load_vocab(vocab_path):
    vocab = collections.OrderedDict()
    with open(vocab_path, "r", encoding='utf-8') as reader:
        for index,line in enumerate(reader):
            token = convert_to_unicode(line[:-1])
            if not token:
                continue
            vocab[token] = index
    print("Successfully Loaded Vocabulary from {}".format(vocab_path))
    return vocab


def find_nearest(indexes, position, max_length=128):
    if position is None or len(indexes) == 1:
        return indexes[0]
    min_diff, target_i = max_length, 0
    for i,index in enumerate(indexes):
        if index < position:
            diff = position - index
        else:
            diff = index - position
        if diff < min_diff:
            min_diff = diff
            target_i = i
    return indexes[target_i]





# tokenizers
class BasicTokenizer(object):
    def __init__(self, do_lower_case=True):
        self.do_lower_case = do_lower_case
    
    def tokenize(self, text):
        text = convert_to_unicode(text)
        text = self._clean_text(text)
        
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)  # list of whitespace splitted tokens
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)  # str removed with accent tokens
            split_tokens.extend(self._run_split_on_punc(token))
    
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens  # list of tokens without whitespaces

    def _run_strip_accents(self, text):
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
              if start_new_word:
                  output.append([])
              start_new_word = False
              output[-1].append(char)
            i += 1
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True
        return False

    def _clean_text(self, text):
        cleaned_text = ""
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                cleaned_text += " "
            else:
                cleaned_text += char
        return cleaned_text




class WordpieceTokenizer(object):
    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        text = convert_to_unicode(text)
    
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue
            
            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                  
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end
    
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens



class FullTokenizer(object):
    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
    
    def tokenize_for_inputs(self, text_a, text_b=None, max_length=512, do_padding=True):
        split_tokens_a = self.tokenize(text_a)
        ids_a = self.convert_tokens_to_ids(split_tokens_a)
        segment_pos = []
        if text_b is None:
            target_length = max_length - 2
            ids_a = ids_a[: target_length]
            input_ids = [CLS_ID] + ids_a + [SEP_ID]
            segment_pos.append(len(input_ids))
        else:
            target_length = max_length - 3
            split_tokens_b = self.tokenize(text_b)
            ids_b = self.convert_tokens_to_ids(split_tokens_b)
            
            while len(ids_a) + len(ids_b) > target_length:
                if len(ids_a) > len(ids_b):
                    if random.random() < 0.5:
                        ids_a.pop()
                    else:
                        ids_a.pop(0)
                else:
                    if random.random() < 0.5:
                        ids_b.pop()
                    else:
                        ids_b.pop(0)
                    
            input_ids = [CLS_ID] + ids_a + [SEP_ID]
            segment_pos.append(len(input_ids))
            input_ids += ids_b + [SEP_ID]
            segment_pos.append(len(input_ids))
        
        input_ids, mlm_labels = self.mask(input_ids)
        
        if do_padding:
            while len(input_ids) < max_length:
                input_ids.append(PAD_ID)
        return input_ids, mlm_labels, segment_pos
        
    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
          for sub_token in self.wordpiece_tokenizer.tokenize(token):
            split_tokens.append(sub_token)
        return split_tokens
    
    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            ids.append(self.vocab.get(token, UNK_ID))
        return ids

    def convert_ids_to_tokens(self, ids):
        tokens = []
        for token_id in ids:
            tokens.append(self.inv_vocab[token_id])
        return tokens

    def mask(self, input_ids):
        mlm_labels = []
        for i,item in enumerate(input_ids):
            if item == CLS_ID or item == SEP_ID:
                continue
            if random.random() < 0.15:
                mlm_labels.append((i, item))
                prob = random.random()
                if prob < 0.8:
                    input_ids[i] = MASK_ID
                elif prob < 0.9:
                    replace_id = random.randint(MASK_ID+1, len(self.vocab)-1)
                    input_ids[i] = replace_id
        return input_ids, mlm_labels




class CKETokenizer(FullTokenizer):    
    def tokenize_for_eval(self, query, answer, position=None, max_length=128, do_padding=True):
        answer_tokens = self.tokenize(answer)
        answer_id = self.convert_tokens_to_ids(answer_tokens)
        if len(answer_id) > 1 or answer_id[0] == UNK_ID:
            return None
        answer_id = answer_id[0]
        
        query_tokens = self.tokenize(query)
        query_ids = self.convert_tokens_to_ids(query_tokens)
        query_ids = query_ids[: max_length]
        
        indexes = []
        for index,qid in enumerate(query_ids):
            if qid == answer_id:
                indexes.append(index)
        if (len(indexes) == 0) or (len(indexes) > 1 and not position):  
            return None   # target word trimmed after cutting to max_length
        index = find_nearest(indexes, position, max_length)
    
        mlm_labels = [0 for _ in query_ids]
        mlm_labels[index] = query_ids[index]
        query_ids[index] = MASK_ID
        
        attn_mask = [1 for _ in query_ids]
        
        if do_padding:
            while len(query_ids) < max_length:
                query_ids.append(PAD_ID)
                mlm_labels.append(0)
                attn_mask.append(0)
        token_type_ids = [1-x for x in attn_mask]
        return query_ids, token_type_ids, attn_mask, mlm_labels




class BertTokenizer(object):
    def __init__(self, args, do_lower_case=True, max_len=None, do_basic_tokenize=True):
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer()
        self.wordpiece_tokenizer = WordpieceTokenizer()
        self.max_len = max_len if max_len else int(1e12)

    def tokenize(self, text, vocab):
        if self.do_basic_tokenize:
            split_tokens = []
            basic_tokens = self.basic_tokenizer.tokenize(text)
            for token in basic_tokens:
                split_tokens.extend(self.wordpiece_tokenizer.tokenize(token, vocab))
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text, vocab)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab.w2i[token])
        if len(ids) > self.max_len:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.vocab.i2w[i])
        return tokens
