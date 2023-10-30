import string
import ast

import torch
from torch.utils import data
from torch.utils.data import DataLoader
import json


class qa_dataset(data.Dataset):
    def __init__(self, data_dict):
        super(qa_dataset, self).__init__()
        self.data = data_dict
        self.begin_token = 'BEG'
        self.end_token = 'END'
        # print this out to check valid chars

        printable_chars = string.printable
        self.vocab = []
        for id in range(len(printable_chars)):
            self.vocab.append(printable_chars[id])
        # the last token is the END of the answer or question
        self.vocab.append(self.begin_token)
        self.vocab.append(self.end_token)
        self.vocab_size = len(self.vocab)
        print('Vocabulary has %d tokens' % (self.vocab_size,))
        print(self.vocab)
        self.pos_beg_token = self.vocab.index(self.begin_token)
        self.pos_end_token = self.vocab.index(self.end_token)
        # print(self.pos_end_token)

    def __len__(self):
        """__len__"""
        return len(self.data)

    def __get_str_tensor__(self, input_str):
        # string convert to char index tensor
        tensor = torch.zeros(len(input_str) + 2).long()
        for c in range(len(input_str)):
            try:
                tensor[c + 1] = self.vocab.index(input_str[c])
            except:
                # ignore non-existed char in vocab
                continue
        # add [BEG] token to the head of question / answer string
        # append [END] token to the end of question / answer string
        tensor[0] = self.pos_beg_token
        tensor[-1] = self.pos_end_token
        return tensor

    def get_tensor(self, input_str):
        tensor = self.__get_str_tensor__(input_str)
        return tensor

    def get_char(self, index):
        return self.vocab[index]

    def get_beg_tensor(self):
        tensor = torch.tensor([self.pos_beg_token]).long()
        return tensor

    def __getitem__(self, index: int):
        query_str = self.data[index]['question'].strip()

        # Each query has several candidate answers with counts
        answer_choice = ast.literal_eval(self.data[index]['answers'])
        keys = [k for k, n in answer_choice.items()]
        num_samples = torch.tensor([n for k, n in answer_choice.items()]).float()

        # We use counts as probability to select each answer
        id2 = torch.multinomial(num_samples, 1)[0]
        answer_str = keys[id2].strip()

        # Encode answer string to char tensor
        char_tensors = []
        for tmp_str in [query_str, answer_str]:
            tensor = self.__get_str_tensor__(tmp_str)
            char_tensors.append(tensor)

        return char_tensors
