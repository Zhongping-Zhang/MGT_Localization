import os
import random
import torch
import nltk
import ujson as json
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloaders.dataloader_original_articles import Visualnews_article
from dataloaders.dataloader_utils import load_base_model_tokenizer, merge_sentences, DEVICE, ROBERTA_MAX_TEXT_LENGTH, GPT2_MAX_TEXT_LENGTH
print("DEVICE:", DEVICE)

def sample_from_model(texts, base_model, base_tokenizer,
                      do_top_p=True, top_p=0.94, do_top_k=False, top_k=10,
                      mani_text_min_length=40, mani_text_max_length=300,
                      segment_num=2,
                      prompt_length=5, # number of prompt sentences (split by ".")
                      ):
    # encode each text as a list of token ids

    """predefine parameters for each sampled article"""
    original_sentences = nltk.sent_tokenize(texts[0]) # paragraph_replace_idx = random.randint(1, len(paragraphs)) # 0: title & first paragraph, sample from index 1, the paragraph to replace
    sample_text_length = random.randint(mani_text_min_length, mani_text_max_length)

    prompt_text_original = ". ".join(texts[0].split(". ")[:prompt_length])+"."
    prompt_encoded = base_tokenizer([prompt_text_original], return_tensors="pt", padding=True, truncation=True, max_length=GPT2_MAX_TEXT_LENGTH-sample_text_length).to(DEVICE)
    prompt_text = base_tokenizer.decode(prompt_encoded['input_ids'][0], skip_special_tokens=True) # keys: ['input_ids', 'attention_mask'], constrained by the maximum length of GPT models
    prompt_token_length = prompt_encoded['input_ids'].shape[-1]


    sampling_kwargs = {}
    if do_top_p:
        sampling_kwargs['top_p'] = top_p
    elif do_top_k:
        sampling_kwargs['top_k'] = top_k
    min_length = sample_text_length
    outputs = base_model.generate(**prompt_encoded, min_length=min_length,
                                  max_length=prompt_token_length+sample_text_length,
                                  do_sample=True, **sampling_kwargs,
                                  pad_token_id=base_tokenizer.eos_token_id,
                                  eos_token_id=base_tokenizer.eos_token_id)

    decoded = base_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    generated_text = decoded[0][len(prompt_text):]
    generated_sentences = nltk.sent_tokenize(generated_text)
    try:
        if generated_sentences[-1][-1]!="." and len(generated_sentences)>segment_num: # remove the final sentences if it's incomplete
            generated_sentences=generated_sentences[:-1]
    except:
        print("generated_sentences", generated_sentences)

    mixed_sentences, mixed_labels, segment_num_revise = merge_sentences(original_sentences, generated_sentences, segment_num=segment_num)

    # print("sampled text length:", sample_text_length)
    config_dict = {
        'mixed_labels': mixed_labels,
        'number_of_chunks': segment_num_revise,
        'sample_token_length': sample_text_length,
    } # dictionary to save configurations

    return original_sentences, mixed_sentences, config_dict



if __name__=="__main__":
    import numpy as np
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default="data/Visualnews/filtereddata/visualnews_test.jsonl",
                        help='path to goodnews file')
    parser.add_argument('--model_name', type=str, default="gpt2", help="language model")
    parser.add_argument('--cache_dir', type=str, default="/projectnb/ivc-ml/zpzhang/checkpoints/transformers_cache")
    parser.add_argument('--output_dir', type=str, default="data/Text_Localization/Visualnews")
    parser.add_argument('--article_num', type=int, default=5,) # set number of article to 5 for DEBUG purpose
    parser.add_argument('--segment_num',type=int, default=2,)
    parser.add_argument('--random_seed', type=int, default=0, )
    args = parser.parse_args()
    print(args)

    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    if "visualnews" in args.file_path:
        news_data = Visualnews_article(file_path=args.file_path, article_num=args.article_num)
        news_dataloader = DataLoader(news_data, batch_size=1, shuffle=False)

    base_model, base_tokenizer = load_base_model_tokenizer(model_name=args.model_name, cache_dir=args.cache_dir)

    batch_id = 0
    all_samples = []
    for batch in tqdm(news_dataloader, desc="manipulate Visualnews articles"):
        article_id, text = batch
        article_id = article_id[0]

        sample_kwargs = {}
        if np.random.uniform(0, 1) > 0.5:
            sample_kwargs['do_top_p'] = True
            sample_kwargs['do_top_k'] = False
        else:
            sample_kwargs['do_top_p'] = False
            sample_kwargs['do_top_k'] = True
        sample_kwargs['top_p'] = 0.96
        sample_kwargs['top_k'] = 40

        original_sentences, mixed_sentences, config_dict = sample_from_model(texts=text,
                                                                      base_model=base_model,
                                                                      base_tokenizer=base_tokenizer,
                                                                      segment_num=args.segment_num,
                                                                      **sample_kwargs)

        config_dict.update(sample_kwargs)

        # update info
        config_dict['model_name'] = args.model_name
        sample_kwargs['article_id'] = article_id

        sample_dict = {'article_id': article_id,
                       'original_sentences': original_sentences,
                       'merge_sentences': mixed_sentences,
                       'config_dict': config_dict,
                       }
        all_samples.append(sample_dict)

        batch_id += 1
        if batch_id > args.article_num:
            break

        os.makedirs(args.output_dir, exist_ok=True)



    model_name = args.model_name.replace("/", "_")
    data_info = os.path.basename(args.file_path).split(".")[0]
    # save data into json file
    with open(os.path.join(args.output_dir, "%s-%s-art%d-seg%d.json"%(data_info, model_name, args.article_num, args.segment_num)), 'w', encoding='utf-8') as file:
        json.dump(all_samples, file, indent=2)










