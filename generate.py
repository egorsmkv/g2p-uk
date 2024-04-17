import torch
import torch.nn.functional as F
import numpy as np

from collections import Counter, defaultdict

class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = defaultdict(int)
        self.total = 0

    def add_word(self, word, freq=1):
        try:
            token_id = self.word2idx[word]
        except KeyError:
            self.idx2word.append(word)
            token_id = self.word2idx[word] = len(self.idx2word) - 1
        if freq:
            self.counter[token_id] += freq
            self.total += freq
        return token_id

    def update(self, words: Counter):
        for word in words:
            self.add_word(word, freq=0)
        self.counter.update(words)
        self.total += words.total()

    def __len__(self):
        return len(self.idx2word)

def produce_vocab_logits(head_weight, head_bias, hiddens):
    return torch.nn.functional.linear(hiddens, head_weight, bias=head_bias)

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1 

    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits

# Inference

def convert_words_to_phonemes(words, model, dictionary, cuda):
    n_words = 128
    temperature = 1.0

    phonemized_words = []

    context = 'постійно\tp o s t1 i1 j n o\n'
    context = 'карликом-наркоторговцем\tk a1 r l y k o m n a r k o t o r h o1 v ts e m\n'
    context = 'карасем\tk a r a s e1 m\n'

    hidden = None
    mems = None

    for input_word in words:
        prompt = context + input_word + '\t'
        buffer = np.array(np.frombuffer(prompt.encode('utf-8'), dtype=np.uint8))
        input = torch.from_numpy(buffer).clone().long()[:, None]

        if cuda:
            input = input.cuda()

        logits, hidden, mems = model(input[:-1, :], hidden, mems=mems, return_h=False)
        input = input[-1:, :]

        phonemes = []
        for _ in range(n_words):
            with torch.no_grad():
                logits, hidden, mems = model(input, hidden, mems=mems, return_h=False)

            output = produce_vocab_logits(model.decoder.weight, model.decoder.bias, logits) / temperature
            output = top_k_top_p_filtering(output.view(-1), top_k=1).view(*output.shape)

            word_weights = F.softmax(output, dim=-1).squeeze()

            word_idx = torch.multinomial(word_weights, num_samples=1)[0]
            input.data.fill_(word_idx)
            word = dictionary.idx2word[word_idx]

            phonemes.append(word)

            if word == b'\n':
                break

        phonemized_words.append({
            'graphemes': input_word,
            'phonemes': ''.join([phoneme.decode('utf-8') for phoneme in phonemes])
        })
    
    return phonemized_words

checkpoint = './PRON4.pt'
cuda = False
device = torch.device('cuda' if cuda else 'cpu')

model, _ = torch.load(checkpoint, map_location=device)
model.eval()

if cuda:
    model.cuda()
    model.float()
else:
    model.cpu()

dictionary = Dictionary()
for byte in range(256):
    dictionary.add_word(byte.to_bytes(1, byteorder='little'), freq=0)

words = [
    'постійно',
    'карасем',
]

phonemized_words = convert_words_to_phonemes(words, model, dictionary, cuda)

for word in phonemized_words:
    print('Graphemes:', word['graphemes'])
    print('Phonemes:', word['phonemes'].strip())
    print('---')
