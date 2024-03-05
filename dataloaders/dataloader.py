import os
import random
import torch
import ujson as json
import numpy as np
from typing import Dict
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Sample_Sentence_from_Article(Dataset):
    def __init__(self, file_path="data/Text_Localization/Goodnews/goodnews_test-gpt2-xl-art1000-seg3.json",
                 sentences_in_window=3,
                 n_sample=1000,
                 ):
        self.file_path = file_path
        self.sentences_in_window = sentences_in_window
        self.n_sample = n_sample
        with open(file_path,"r") as f:
            self.article_data = json.load(f)
        self.get_sentence_data() # get self.sentence_data, self.contain_mgt_list
        self.mgt_rate = sum(self.contain_mgt_list)/len(self.contain_mgt_list)

    def get_sentence_data(self):
        samples_per_article = self.n_sample//len(self.article_data)
        samples_per_article = max(1, samples_per_article)

        self.sentence_data = []
        self.contain_mgt_list = []

        remaining_n_sample = self.n_sample
        while remaining_n_sample>0:
            for sample in tqdm(self.article_data,desc="sampling sentences from articles"):
                article_length = len(sample['merge_sentences'])
                if article_length<self.sentences_in_window:
                    continue
                for j in range(samples_per_article):
                    example = self.sample_from_article(sample)
                    contain_mgt = 1 if 1 in example['label'] else 0
                    self.sentence_data.append(self.sample_from_article(sample))
                    self.contain_mgt_list.append(contain_mgt)
                remaining_n_sample-=samples_per_article
                if remaining_n_sample<=0:
                    break


    def sample_from_article(self, sample):
        """designed for self.get_sentence_data"""
        merge_sentences = sample['merge_sentences']
        sentence_label = sample['config_dict']['mixed_labels']
        random_num = random.randint(0, len(merge_sentences)-self.sentences_in_window)


        return_sentence_list = merge_sentences[random_num:random_num+self.sentences_in_window]
        return_label = sentence_label[random_num:random_num+self.sentences_in_window]
        return {"article_id": sample['article_id'],
                # "original_article": merge_sentences,
                # "original_article_label": sentence_label,
                "input_sentences_list": return_sentence_list,
                "input_sentences": " ".join(return_sentence_list),
                "label": return_label,
                "label_np": np.array(return_label),}

    def __len__(self):
        return len(self.sentence_data)

    def __getitem__(self, i):
        sentence_sample = self.sentence_data[i]
        return sentence_sample


def load_tokenizer(model_name, cache_dir):
    optional_tok_kwargs = {}
    if "facebook/opt-" in model_name:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    tokenizer = AutoTokenizer.from_pretrained(model_name, **optional_tok_kwargs, cache_dir=cache_dir)
    return tokenizer

if __name__=="__main__":
    import numpy as np
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default="data/Text_Localization/Goodnews/goodnews_test-gpt2-xl-art1000-seg3.json",
                        help='path to goodnews file')
    parser.add_argument('--model_name', type=str, default="gpt2-xl", help="language model")
    parser.add_argument('--tokenizer_name',type=str, default="roberta-large-openai-detector")
    parser.add_argument('--cache_dir', type=str, default="/projectnb/ivc-ml/zpzhang/checkpoints/transformers_cache")
    parser.add_argument('--article_num', type=int, default=1000000,)
    args = parser.parse_args()
    print(args)


    goodnews_sampled_data = Sample_Sentence_from_Article(file_path=args.file_path, sentences_in_window=1, n_sample=1500)

    sample = goodnews_sampled_data[1]
    print("complete")
















