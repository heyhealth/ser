import os
import torchtext
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
from transformers.models.wav2vec2 import Wav2Vec2CTCTokenizer
from main.opts import ARGS


def sentence_to_tokens(sentence):
    """Converts a sentence to tokens list"""
    tokenizer = get_tokenizer("basic_english")
    tokens = tokenizer(sentence)
    return tokens


def build_vocab_from_iterator_wrapper(tokens_iter):
    """build a vocab by the transcripts token iterator"""

    def yield_tokens(tokens_iter):
        for tokens in tokens_iter:
            yield tokens

    vocab = build_vocab_from_iterator(yield_tokens(tokens_iter), specials=["<unk>"])
    return vocab


def test():
    token_iter = [
        ['mmm', 'hmm', '.', 'i', 'never', 'knew', 'it', 'was', 'going', 'to', 'be', 'this', 'hard', ',', 'you', 'know',
         '.'], ['i', 'know', '.', 'he', 'was', ',', 'he', 'had', 'so', 'much', 'going', 'for', 'him', '.'],
        ['because', 'this', 'other', 'half', 'was', 'just', 'kind', 'of', 'like', 'ripped', 'away', '.']]

    vocab = build_vocab_from_iterator_wrapper(token_iter)

    print(len(vocab))
    print(vocab.lookup_tokens([i for i in range(32)]))
    print(vocab.lookup_indices(['i', 'know', '<unk>']))



def tokenize_for_ctc(text):
    text = text.upper()

    ctc_tokenizer = Wav2Vec2CTCTokenizer(vocab_file=os.path.join(ARGS.PROJECTION_PATH, 'save', 'data', 'vocab.json'))
    # https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/vocab.json
    """
    {'input_ids': [22, 8, 16, 4, 7, 13, 5, 4, 21, 8, 8, 14, 4, 24, 8, 22, 3], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    """
    return ctc_tokenizer(text)


if __name__ == '__main__':
    print(tokenize_for_ctc("You ara good BOy"))
