"""Contains classes for encoding tokens."""

import torch
import torch.nn as nn
import transformers
from transformers import AutoModel, AutoConfig, AutoTokenizer
from abc import ABC, abstractmethod


class BaseEncoder(nn.Module, ABC):
    """BaseEncoder class.

    Args:
        conf (dict): Dictionary with configuration parameters.
        vocab (Vocab): Class instance from vocab.py.
    """
    def __init__(self, conf, vocab):
        super().__init__()
        self.conf = conf
        self.vocab = vocab

    @abstractmethod
    def forward(self, **kwargs):
        raise NotImplementedError('forward method not implemented')


# for other languages (not tested)
class RoBERTaEncoder(BaseEncoder):
    """XLM-RoBERTa encoder."""

    def __init__(self, conf, vocab):
        super().__init__(conf, vocab)
        huggingface_config = transformers.XLMRobertaConfig(pad_token_id=0, bos_token_id=1, eos_token_id=2)
        self.encoder = transformers.XLMRobertaModel(huggingface_config)
        self.linear = nn.Linear(in_features=huggingface_config.hidden_size,
                                out_features=self.conf['grammeme_LSTM_hidden'])

    def forward(self, words_batch, **kwargs):
        # words_batch has shape (max_sentence_length, batch_size)
        input_ids = words_batch.permute(1, 0)
        attention_mask = input_ids.where(input_ids != 0, 0)
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_output = encoder_output.last_hidden_state.flatten(start_dim=0, end_dim=1)
        encoder_output = self.linear(encoder_output)
        return encoder_output


class RuBERTEncoder(BaseEncoder):
    """RuBERT encoder."""

    def __init__(self, conf, vocab):
        super().__init__(conf, vocab)
        self.tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased', pad_token_id=0)
        huggingface_config = transformers.AutoConfig.from_pretrained('DeepPavlov/rubert-base-cased',
                                                                     pad_token_id=0)
        self.encoder = transformers.AutoModel.from_pretrained('DeepPavlov/rubert-base-cased', config=huggingface_config)
        self.linear = nn.Linear(in_features=huggingface_config.hidden_size,
                                out_features=self.conf['grammeme_LSTM_hidden'])

    def forward(self, words_batch, **kwargs):
        tokenized = self.tokenizer(words_batch, is_split_into_words=True, padding=True,
                                   return_tensors='pt', return_attention_mask=True).to(self.conf['device'])
        encoder_output = self.encoder(input_ids=tokenized['input_ids'], attention_mask=tokenized['attention_mask'])
        encoder_output = self.linear(encoder_output.last_hidden_state)
        max_sentence_length = max(map(len, words_batch))
        encoded_subtokens = []
        for i in range(tokenized['input_ids'].shape[0]):
            subtokens_ids = tokenized.word_ids(
                batch_index=i # [None, 0, 1, 1, 1, 2, 3, ..., [ntokens], None, None, None]
            )
            first_subtokens_ids = []
            previous = None
            for j, subtoken_id in enumerate(subtokens_ids):
                if subtoken_id is None or subtoken_id == previous:
                    continue
                else:
                    first_subtokens_ids.append(j)
                    previous = subtoken_id
            tensor_i = encoder_output[i][first_subtokens_ids]
            tensor_i_padding = (torch.zeros(max_sentence_length - tensor_i.shape[0], tensor_i.shape[1])
                                .to(self.conf['device']))
            tensor_i = torch.cat((tensor_i, tensor_i_padding))
            encoded_subtokens.append(tensor_i)

        encoded_subtokens = torch.stack(encoded_subtokens)
        encoded_subtokens = encoded_subtokens.flatten(start_dim=0, end_dim=1)

        return encoded_subtokens
