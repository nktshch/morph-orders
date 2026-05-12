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


class LSTMEncoder(BaseEncoder):
    """Class creates embeddings for words and chars, puts them through LSTMs to produce word embeddings.

    Method forward takes words_batch -- size (max_sentence_length, batch_size) -- and
    chars_batch -- size (max_word_length, max_sentence_length * batch_size) -- as an input. It returns output from the
    LSTM for every word in a sentence.
    The final shape of the output is (max_sentence_length, batch_size, grammeme_LSTM_hidden).

    Args:
        conf (dict): Dictionary with configuration parameters.
        vocab (Vocab): Class instance from vocab.py.
    """

    def __init__(self, conf, vocab):
        super().__init__(conf, vocab)
        self.embeddings = self.vocab.embeddings
        # from_pretrained outputs only torch.float64
        self.word_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(self.embeddings), freeze=False).float()
        self.char_embeddings = nn.Embedding(len(self.vocab.vocab['char-index']), self.conf["char_embeddings_dimension"])
        nn.init.xavier_uniform_(self.char_embeddings.weight)
        self.charLSTM = nn.LSTM(input_size=self.conf['char_embeddings_dimension'],
                                hidden_size=self.conf['char_LSTM_hidden'],
                                bidirectional=self.conf['char_LSTM_bidirectional'],
                                batch_first=False)
        for name, param in self.charLSTM.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
        self.wordLSTMcell = nn.LSTMCell(input_size=(self.conf['word_embeddings_dimension'] +
                                                    self.conf['char_LSTM_hidden'] * self.conf['char_LSTM_directions']),
                                        hidden_size=self.conf['word_LSTM_hidden'])
        for name, param in self.wordLSTMcell.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
        self.wordDropout_input = nn.Dropout(p=self.conf['word_LSTM_input_dropout'])
        self.wordDropout_state = nn.Dropout(p=self.conf['word_LSTM_state_dropout'])
        self.wordDropout_output = nn.Dropout(p=self.conf['word_LSTM_output_dropout'])

    def forward(self, words_batch, chars_batch, oov=None, **kwargs):
        """Takes batches of indices of words and chars and creates embeddings with LSTM.

        PyTorch LSTM module doesn't return cell states by default. That is why we have to use LSTMCell in a loop.

        Args:
            words_batch (torch.Tensor): Tensor of words indices for every word in a batch.
                Size (max_sentence_length, batch_size).
            chars_batch (torch.Tensor): Tensor of chars indices for every word in a batch.
                Size (max_word_length, max_sentence_length * batch_size).
            oov (tuple): Out of vocab embeddings that are used during inference.

        Returns:
            tuple: Tuple consists of two tensors - one with hidden states, and one with cell states of the word LSTM.
                The shape of each tensor is (max_sentence_length * batch_size, grammeme_LSTM_hidden).
        """

        current_batch_size = words_batch.shape[1]
        words = self.word_embeddings(words_batch)
        if oov is not None:
            fasttext_embeddings, mask_embeddings = oov
            words[mask_embeddings] = fasttext_embeddings
        chars = self.char_embeddings(chars_batch)
        # words has shape (max_sentence_length, batch_size, word_embeddings_dimension)
        # chars has shape (max_word_length, max_sentence_length * batch_size, char_embeddings_dimension)
        _, (hn, cn) = self.charLSTM(chars)
        # hn has shape (char_LSTM_directions, max_sentence_length * batch_size, char_LSTM_hidden)

        if hn.shape[0] == 1:
            chars = hn[0].reshape(current_batch_size, -1, hn.shape[2] * hn.shape[0]).permute(1, 0, 2)
        else:
            chars = torch.concat((hn[0], hn[1]), dim=1)
            chars = chars.reshape(current_batch_size, -1, hn.shape[2] * hn.shape[0]).permute(1, 0, 2)
        # note: it's impossible to avoid using permute because we can't control how reshape splits original tensor

        # chars has shape (max_sentence_length, batch_size, char_LSTM_directions * char_LSTM_hidden)

        words = torch.concat((words, chars), dim=2)
        words = self.wordDropout_input(words)
        # words has shape
        # (max_sentence_length, batch_size, word_embeddings_dimension + char_LSTM_directions * char_LSTM_hidden)

        hidden_forward, cell_forward = self.loop(words)

        if self.conf['word_LSTM_bidirectional']:
            hidden_backward, cell_backward = self.loop(words.flip(dims=[0]))

            hidden_backward = hidden_backward.flip(dims=[0])
            cell_backward = cell_backward.flip(dims=[0])

            hidden = torch.concat((hidden_forward, hidden_backward), dim=2)
            cell = torch.concat((cell_forward, cell_backward), dim=2)

            hidden = hidden.permute(1, 0, 2).reshape(-1, hidden.size(dim=2))
            cell = cell.permute(1, 0, 2).reshape(-1, cell.size(dim=2))
            # again, impossible to use only reshape
            hidden = self.wordDropout_output(hidden)
            cell = self.wordDropout_output(cell)
            return hidden, cell # shape (max_sentence_length * batch_size, grammeme_LSTM_hidden)

        hidden = hidden_forward
        cell = cell_forward

        hidden = hidden.permute(1, 0, 2).reshape(-1, hidden.size(dim=2))
        cell = cell.permute(1, 0, 2).reshape(-1, cell.size(dim=2))
        # again, impossible to use only reshape
        hidden = self.wordDropout_output(hidden)
        cell = self.wordDropout_output(cell)
        return hidden, cell # shape (max_sentence_length * batch_size, grammeme_LSTM_hidden)

    def loop(self, words):
        hk = torch.zeros((words.size(dim=1), self.wordLSTMcell.hidden_size)).to(self.conf['device'])
        ck = torch.zeros((words.size(dim=1), self.wordLSTMcell.hidden_size)).to(self.conf['device'])
        hidden = []
        cell = []
        for word in words:
            hk = self.wordDropout_state(hk)
            ck = self.wordDropout_state(ck)
            hk, ck = self.wordLSTMcell(word, (hk, ck))
            hidden += [hk]
            cell += [ck]
        hidden = torch.stack(hidden)
        cell = torch.stack(cell)

        return hidden, cell


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
