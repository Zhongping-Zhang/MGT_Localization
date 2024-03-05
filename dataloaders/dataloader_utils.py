import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import pandas as pd
import tqdm
# import nltk
random.seed(2024)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROBERTA_MAX_TEXT_LENGTH=512 # maximum text length: 512
GPT2_MAX_TEXT_LENGTH=1024

def load_base_model_tokenizer(model_name,cache_dir):
    """load the language models and their corresponding tokenizers"""
    base_model_kwargs = {}
    if 'gpt-j' in model_name or 'neox' in model_name:
        base_model_kwargs.update(dict(torch_dtype=torch.float16))
    if 'gpt-j' in model_name:
        base_model_kwargs.update(dict(revision='float16'))
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **base_model_kwargs,
        cache_dir=cache_dir
    ).to(DEVICE)
    optional_tok_kwargs = {}
    if "facebook/opt-" in model_name:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    base_tokenizer = AutoTokenizer.from_pretrained(model_name, **optional_tok_kwargs, cache_dir=cache_dir)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
    return base_model, base_tokenizer


def merge_sentences(original_sentences, generated_sentences, segment_num=2):
    if len(generated_sentences)==0:
        print("no generated sentences")
        return original_sentences, [0 for i in range(len(original_sentences))], 0

    if len(generated_sentences)<segment_num:
        # print("generated_sentences length:", len(generated_sentences))
        segment_num=len(generated_sentences) # in this case, segment_size=1, fix the bug for separate characters
        segment_size = 1
    else:
        segment_size = len(generated_sentences)//segment_num
    divided_generated_sentences = []
    for i in range(segment_num):
        if i==segment_num-1:
            divided_generated_sentences.append(generated_sentences[i*segment_size:])
        else:
            divided_generated_sentences.append(generated_sentences[i*segment_size:(i+1)*segment_size])

    total_len = len(original_sentences)+len(generated_sentences)
    try:
        random_positions = random.sample(range(1,len(original_sentences)), segment_num)
    except:
        print("length of original sentences:",len(original_sentences))
        print("number of segment:", segment_num)
        if len(original_sentences)==1:
            segment_num=1
            random_positions=[1]
        else:
            segment_num = len(original_sentences)-1
            random_positions=random.sample(range(1,len(original_sentences)), segment_num)

    return_sentences = []
    return_labels = []

    segment_id = 0
    for sentence_id in range(len(original_sentences)):
        if sentence_id not in random_positions:
            return_sentences.append(original_sentences[sentence_id])
            return_labels.append(0) # 0 represents human-written text
        else:
            return_sentences+=divided_generated_sentences[segment_id]
            return_labels+=[1]*len(divided_generated_sentences[segment_id])
            segment_id+=1

    return return_sentences, return_labels, segment_num


def get_roberta_feature(sample, roberta_detector, roberta_tokenizer):
    sample_article_id = sample["article_id"]
    sample_original_article = "\n\n".join(sample["original_paragraphs"])
    sample_manipulated_article = "\n\n".join(sample["merge_paragraphs"])
    sample_config_dict = sample["config_dict"]
    sample_manipulated_article_token = roberta_tokenizer(sample_manipulated_article,
                                                         padding='max_length',  # longest, max_length, False
                                                         truncation=True,
                                                         max_length=512,
                                                         return_tensors="pt").to(DEVICE)  # (1, text_length), text_length should be smaller than 512
    sample_manipulated_article_embeddings = roberta_detector(**sample_manipulated_article_token,
                                                             output_hidden_states=True, return_dict=True)
    last_hidden_state = sample_manipulated_article_embeddings['hidden_states'][-1]  # (1,512,1024)
    return last_hidden_state









