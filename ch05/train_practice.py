# pip install torch torchtext
import torch
import torchtext
from torchtext.datasets import PennTreebank
torchtext.disable_torchtext_deprecation_warning()

# 데이터 로딩 및 전처리
def load_data(batch_size=32, bptt=35):
    train_dataset, val_dataset, test_dataset = PennTreebank(root='.data', split=('train', 'valid', 'test'))
    
    train_data = list(train_dataset)
    chars = list(set(''.join(train_data)))
    data_size, vocab_size = len(''.join(train_data)), len(chars)
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
    
    return ''.join(train_data), char_to_ix, ix_to_char, data_size, vocab_size

# 데이터 로딩
train_data, char_to_ix, ix_to_char, data_size, vocab_size = load_data()

import numpy as np

# 하이퍼파라미터 설정
hidden_size = 100
seq_length = 25
learning_rate = 1e-1

# RNN 모델 클래스
class RNN:
    def __init__(self, hidden_size, vocab_size):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # 가중치 초기화
        self.Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(vocab_size, hidden_size) * 0.01
        
        # 편향 초기화
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))
        
    def forward(self, inputs, h_prev):
        h = np.tanh(self.Wxh @ inputs + self.Whh @ h_prev + self.bh)
        y = self.Why @ h + self.by
        return y, h
    
    def loss(self, inputs, targets, h_prev):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        loss = 0
        
        # 순전파
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1
            ys[t], hs[t] = self.forward(xs[t], hs[t-1])
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            loss += -np.log(ps[t][targets[t], 0])
        
        # 역전파
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dh_next = np.zeros_like(hs[0])
        
        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1
            dWhy += dy @ hs[t].T
            dby += dy
            dh = self.Why.T @ dy + dh_next
            dhraw = (1 - hs[t] * hs[t]) * dh
            dbh += dhraw
            dWxh += dhraw @ xs[t].T
            dWhh += dhraw @ hs[t-1].T
            dh_next = self.Whh.T @ dhraw
        
        # 기울기 클리핑
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        
        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]
    
    def sample(self, h, seed_ix, n):
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        
        for t in range(n):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            y = self.Why @ h + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
        
        return ixes

# 모델 학습
rnn = RNN(hidden_size, vocab_size)
n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(rnn.Wxh), np.zeros_like(rnn.Whh), np.zeros_like(rnn.Why)
mbh, mby = np.zeros_like(rnn.bh), np.zeros_like(rnn.by)
smooth_loss = -np.log(1.0/vocab_size)*seq_length

while True:
    if p+seq_length+1 >= len(train_data) or n == 0:
        h_prev = np.zeros((hidden_size,1))
        p = 0
    
    inputs = [char_to_ix[ch] for ch in train_data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in train_data[p+1:p+seq_length+1]]
    
    loss, dWxh, dWhh, dWhy, dbh, dby, h_prev = rnn.loss(inputs, targets, h_prev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    
    if n % 1000 == 0:
        print(f'iter {n}, loss: {smooth_loss}')
        sample_ix = rnn.sample(h_prev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n %s \n----' % (txt, ))
    
    for param, dparam, mem in zip([rnn.Wxh, rnn.Whh, rnn.Why, rnn.bh, rnn.by], 
                                  [dWxh, dWhh, dWhy, dbh, dby],
                                  [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)
   
    p += seq_length
    n += 1