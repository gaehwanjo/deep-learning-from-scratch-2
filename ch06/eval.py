# coding: utf-8
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)
from rnnlm import Rnnlm
from better_rnnlm import BetterRnnlm
from dataset import ptb
from common.util import eval_perplexity


if __name__ == '__main__':
    model = Rnnlm()
    #model = BetterRnnlm()

    # 학습된 매개변수 읽기
    model.load_params()

    corpus, _, _ = ptb.load_data('test')

    model.reset_state()
    ppl_test = eval_perplexity(model, corpus)
    print('test perplexity: ', ppl_test)
