from pytorch_transformers import *
from pytorch_transformers.tokenization_bert import whitespace_tokenize
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import collections
import string
import re

##########################
######## CLASSES #########
##########################


class RobertaForForQuestionAnswering(BertPreTrainedModel):

    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, start_positions=None, end_positions=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)

        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        # (loss), start_logits, end_logits, (hidden_states), (attentions)
        return outputs


class RNNForForQuestionAnswering(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.qa_outputs = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input_ids, text_lengths, start_positions=None, end_positions=None):
        embedded = self.dropout(self.embedding(input_ids))

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths, batch_first=True, enforce_sorted=False)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True)

        logits = self.qa_outputs(output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits)
        if start_positions is not None and end_positions is not None:
                # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs


class RNNForSequenceClassification(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):

        super().__init__()

        self.num_labels = output_dim

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, text_lengths, labels=None):
        embedded = self.dropout(self.embedding(input_ids))

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths, batch_first=True, enforce_sorted=False)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True)

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout
        hidden = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        logits = self.fc(hidden)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))

        return (loss, logits)  # (loss), logits


class LSTMTokenizer():
    def __init__(self, vocab):
        self.pad_token = "[PAD]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.unk_token = "[UNK]"
        self.vocab = vocab

    def tokenize(self, text):
        text = text.strip()
        if not text:
            return []
        tokens = text.split()
        return tokens

    def convert_tokens_to_ids(self, tokens):
        token_ids = []
        for token in tokens:
            if token not in self.vocab:
                token = self.unk_token
            token_ids.append(self.vocab[token])
        return token_ids

    def convert_tokens_to_string(self, tokens):
        out_string = ' '.join(tokens).strip()
        return out_string


def getLSTM(device, embeddings=None, qa=False):
    INPUT_DIM = 400004
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 384
    OUTPUT_DIM = 2
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    PAD_IDX = 3

    if qa:
        model = RNNForForQuestionAnswering(INPUT_DIM,
                                           EMBEDDING_DIM,
                                           HIDDEN_DIM,
                                           OUTPUT_DIM,
                                           N_LAYERS,
                                           BIDIRECTIONAL,
                                           DROPOUT,
                                           PAD_IDX)
    else:
        model = RNNForSequenceClassification(INPUT_DIM,
                                             EMBEDDING_DIM,
                                             HIDDEN_DIM,
                                             OUTPUT_DIM,
                                             N_LAYERS,
                                             BIDIRECTIONAL,
                                             DROPOUT,
                                             PAD_IDX)

    if embeddings:
        model.embedding.weight.data.copy_(torch.tensor(embeddings))

    model.to(device)
    return model


def getGloveData():
    glove_path = "glove"

    words = ['[CLS]', '[SEP]', '[UNK]', '[PAD]']
    idx = 4
    word2idx = {
        '[CLS]': 0,
        '[SEP]': 1,
        '[UNK]': 2,
        '[PAD]': 3
    }
    vectors = []

    with open(f'{glove_path}/glove.6B.100d.txt', 'rb') as f:
        for token in words:
            vectors.append(np.zeros(100))

        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    glove = {w: vectors[word2idx[w]] for w in words}

    return (glove, vectors, word2idx, words)

##########################
### DATA PREPROCESSING ###
##########################


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def format_df(df):
    doc_tokens_list = []
    questions = []
    answers_text = []
    start_position = []
    end_position = []
    is_impossible = []
    num_possible_answers = []
    gold_answers = []
    for i in range(len(df)):
        topic = df['data'][i]['paragraphs']
        for sub_para in topic:
            paragraph_text = sub_para["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True

            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for q_a in sub_para['qas']:
                if not q_a['is_impossible']:
                    answer_offset = q_a['answers'][0]['answer_start']
                    answer_length = len(q_a['answers'][0]['text'])
                    start_position_value = char_to_word_offset[answer_offset]
                    end_position_value = char_to_word_offset[answer_offset +
                                                             answer_length - 1]

                    # Only add answers where the text can be exactly recovered from the
                    # document. If this can't happen it's likely due to weird unicode
                    # stuff so we will just skip the example.
                    actual_text = " ".join(
                        doc_tokens[start_position_value:(end_position_value + 1)])
                    cleaned_answer_text = " ".join(
                        whitespace_tokenize(q_a['answers'][0]['text']))

                    if actual_text.find(cleaned_answer_text) == -1:
                        logger.warning(
                            "Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                        continue
                    else:
                        # add example
                        start_position.append(start_position_value)
                        end_position.append(end_position_value)
                        answers_text.append(q_a['answers'][0]['text'])
                else:
                    start_position.append(-1)
                    end_position.append(-1)
                    answers_text.append("")

                questions.append(q_a['question'])
                doc_tokens_list.append(doc_tokens)
                is_impossible.append(q_a['is_impossible'])
                num_possible_answers.append(len(q_a['answers']))
                gold_answers.append(
                    list(map(lambda x: x['text'], q_a['answers'])))

    return pd.DataFrame({
        "question": questions,
        "doc_tokens": doc_tokens_list,
        "orig_answer_text": answers_text,
        "start_position": start_position,
        "end_position": end_position,
        "is_impossible": is_impossible,
        "num_possible_answers": num_possible_answers,
        "gold_answers": gold_answers
    })


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start or position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + \
            0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

#########################
###### EVALUATION #######
#########################


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_squad_scores(preds_start, preds_end, all_tokens, all_gold_answers, cls_indexes, tokenizer):
    predictions = []
    f1_score_list = []
    exact_match_list = []
    for i, start_index in enumerate(preds_start):
        end_index = preds_end[i]
        gold_answers = all_gold_answers[i]
        cls_index = cls_indexes[i]
        if len(gold_answers) == 0:
            gold_answers.append("")

        if start_index == cls_index and end_index == cls_index:
            pred_text = ""
        else:
            tokens = all_tokens[i]

            pred_tokens = tokens[start_index:(end_index+1)]
            if tokenizer:
                pred_text = tokenizer.convert_tokens_to_string(pred_tokens)
            else:
                pred_text = " ".join(pred_tokens)

                # De-tokenize WordPieces that have been split off.
                pred_text = pred_text.replace(" ##", "")
                pred_text = pred_text.replace("##", "")

            # Clean whitespace
            pred_text = pred_text.strip()
            pred_text = " ".join(pred_text.split())

        f1_score_list.append(max(compute_f1(a, pred_text)
                                 for a in gold_answers))
        exact_match_list.append(max(compute_exact(a, pred_text)
                                    for a in gold_answers))
    return (np.mean(f1_score_list), np.mean(exact_match_list))
