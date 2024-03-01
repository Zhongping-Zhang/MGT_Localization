# Original work Copyright (c) Guangsheng Bao.
# Modified work Copyright 2024 Zhongping Zhang.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import json
import numpy as np
import torch
import argparse
from gradio_utils.fastdetectgpt_scripts.model import load_tokenizer, load_model
from gradio_utils.fastdetectgpt_scripts.fast_detect_gpt import get_sampling_discrepancy_analytic
# from model import load_tokenizer, load_model
# from fast_detect_gpt import get_sampling_discrepancy_analytic



class ProbEstimator:
    def __init__(self, args):
        self.real_crits = []
        self.fake_crits = []
        os.system("pwd")
        for result_file in glob.glob(os.path.join(args.ref_path, '*.json')):
            # print("result_file:", result_file)
            with open(result_file, 'r') as fin:
                res = json.load(fin)
                self.real_crits.extend(res['predictions']['real'])
                self.fake_crits.extend(res['predictions']['samples'])
        print(f'ProbEstimator: total {len(self.real_crits) * 2} samples.')

    def crit_to_prob(self, crit):
        offset = np.sort(np.abs(np.array(self.real_crits + self.fake_crits) - crit))[100]
        cnt_real = np.sum((np.array(self.real_crits) > crit - offset) & (np.array(self.real_crits) < crit + offset))
        cnt_fake = np.sum((np.array(self.fake_crits) > crit - offset) & (np.array(self.fake_crits) < crit + offset))
        return cnt_fake / (cnt_real + cnt_fake)



# run interactive local inference

def get_fastdetectgpt_score(args, input_text,
    scoring_tokenizer=None, scoring_model=None, reference_tokenizer=None, reference_model=None):
    # load model
    if scoring_model is None:
        scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.dataset, args.cache_dir)
        scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval()

    if reference_model is None:
        if args.reference_model_name != args.scoring_model_name:
            reference_tokenizer = load_tokenizer(args.reference_model_name, args.dataset, args.cache_dir)
            reference_model = load_model(args.reference_model_name, args.device, args.cache_dir)
            reference_model.eval()

    # evaluate criterion
    name = "sampling_discrepancy_analytic"
    criterion_fn = get_sampling_discrepancy_analytic
    prob_estimator = ProbEstimator(args)
    # input text


    text = input_text
    # evaluate text
    tokenized = scoring_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
    labels = tokenized.input_ids[:, 1:]
    with torch.no_grad():
        logits_score = scoring_model(**tokenized).logits[:, :-1]
        if args.reference_model_name == args.scoring_model_name:
            logits_ref = logits_score
        else:
            tokenized = reference_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
            assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
            logits_ref = reference_model(**tokenized).logits[:, :-1]
        crit = criterion_fn(logits_ref, logits_score, labels)
    # estimate the probability of machine generated text
    prob = prob_estimator.crit_to_prob(crit)
    return crit, prob
    # print(f'Fast-DetectGPT criterion is {crit:.4f}, suggesting that the text has a probability of {prob * 100:.0f}% to be fake.')
    # print()


class fastdetectgpt_args:
    def __init__(self, reference_model_name="gpt-neo-2.7B",
                       scoring_model_name="gpt-neo-2.7B",
                       dataset="xsum",
                       ref_path="./local_infer_ref",
                       device="cuda",
                       cache_dir="/projectnb/ivc-ml/zpzhang/checkpoints/transformers_cache"):
        self.reference_model_name=reference_model_name
        self.scoring_model_name=scoring_model_name
        self.dataset=dataset
        self.ref_path=ref_path
        self.device=device
        self.cache_dir=cache_dir

if __name__ == '__main__':
    args = fastdetectgpt_args()
    crit, prob = get_fastdetectgpt_score(args, input_text="Disguised as police, they broke through a fence on Monday evening and broke into the cargo of a Swiss-bound plane to take the valuable items. The audacious heist occurred at an airport in a small European country, leaving authorities baffled and airline officials in shock.")
    print(f'Fast-DetectGPT criterion is {crit:.4f}, suggesting that the text has a probability of {prob * 100:.0f}% to be fake.')






