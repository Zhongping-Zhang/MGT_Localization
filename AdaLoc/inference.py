import transformers
import torch
from tqdm import tqdm
import argparse
import numpy as np
# from localization.detector_utils import find_continuous_intervals
from sklearn.metrics import average_precision_score



DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROBERTA_MAX_TEXT_LENGTH=512
GPT2_MAX_TEXT_LENGTH=1024

def security_check(mixed_sentences):
    single_num=0
    for sent in mixed_sentences:
        if len(sent)<=2:
            single_num+=1

    if single_num>=10:
        return 0
    else:
        return 1


def run_supervised_experiment_sentence_head(data, cache_dir, DEVICE, pos_bit=0, num_labels=2, window_size=3, sentence_head_model=None):
    print(f'Beginning supervised evaluation with sentence head model...')
    tokenizer = sentence_head_model.roberta_tokenizer
    test_preds_list = []
    test_gt_list = []
    AP_list = []
    dataset_single_preds_list = []
    dataset_preds_list = []
    dataset_gt_list = [] # calculate average precision all together
    invalid_num=0

    for sample in tqdm(data, desc="run evaluation on articles"):
        try:  # For GoodNews, VisualNews, WikiText datasets
            article_id = sample['article_id']
            test_mixed_text = " ".join(sample['merge_sentences'])
            test_mixed_sentences = sample['merge_sentences']
            label = sample['config_dict']['mixed_labels']  # sentence label
            num_chunks = sample['config_dict']['number_of_chunks']
            model_name = sample['config_dict']['model_name']
        except:  # For GhostBuster datasets
            test_mixed_sentences = sample['return_sentences']
            label = sample['return_labels']

        valid_label = security_check(test_mixed_sentences)
        if valid_label==0:
            invalid_num+=1
            print("invalid_num", invalid_num)
            continue

        test_preds,_,test_preds_single = get_supervised_model_prediction(
            sentence_head_model, tokenizer, test_mixed_sentences, DEVICE=DEVICE, pos_bit=pos_bit, window_size=window_size)# window_size: how many sentences within a window
        AP = average_precision_score(y_true=np.array(label), y_score=np.array(test_preds))
        # AP_single = average_precision_score(y_true=np.array(label), y_score=np.array(test_preds))

        assert len(test_preds)==len(label), "check label for each article"
        test_preds_list.append(np.array(test_preds))
        test_gt_list.append(np.array(label))
        AP_list.append(AP)

        dataset_preds_list += test_preds
        dataset_gt_list += label
        dataset_single_preds_list += test_preds_single


    results = {
        'prediction': dataset_preds_list,
        'ground_truth': dataset_gt_list,
        'AP_list': AP_list,
        'AP': sum(AP_list)/len(AP_list),
        'dataset_AP': average_precision_score(y_true=np.array(dataset_gt_list), y_score=np.array(dataset_preds_list)),
        # 'dataset_singleAP': average_precision_score(y_true=np.array(dataset_gt_list), y_score=np.array(dataset_single_preds_list)),
    }

    return results




def get_supervised_model_prediction(model, tokenizer, sentence_list, DEVICE, pos_bit=0, window_size=1, window_step=1):
    """
    :param model:
    :param tokenizer:
    :param sentence_list:
    :param DEVICE:
    :param pos_bit:
    :param window_size: model becomes more reliable after around 50 tokens
    :param window_step:
    :return:
    """
    with torch.no_grad():
        # get predictions for real
        preds = []

        each_sample_preds = []
        majority_vote_preds = [[] for i in range(len(sentence_list))]


        for window_start in range(0, max(1, len(sentence_list)-window_size+1), window_step):
            text_data = sentence_list[window_start:window_start+window_size]
            text_merge = " ".join(text_data)
            # window_token_data = tokenizer(text_merge, padding=True, truncation=True,
            #                               max_length=ROBERTA_MAX_TEXT_LENGTH, return_tensors="pt").to(DEVICE)
            # prediction_score = model(**window_token_data).logits.softmax(-1)[:, pos_bit].tolist()[0]
            sentence_feature = model.extract_roberta_feature(text_merge)
            prediction_score = torch.sigmoid(model(sentence_feature)).tolist()[0]

            each_sample_preds.append(prediction_score[1])
            try:
                idx=0
                for vote_idx in range(window_start, window_start+window_size):
                    majority_vote_preds[vote_idx].append(prediction_score[idx])
                    idx+=1
            except:
                idx=0
                for vote_idx in range(window_start, min(window_start+window_size, len(sentence_list))):
                    majority_vote_preds[vote_idx].append(prediction_score[idx])
                    idx+=1


    majority_vote_preds_mean = [sum(sub_list)/len(sub_list) for sub_list in majority_vote_preds]
    each_sample_preds = [majority_vote_preds_mean[0]] + each_sample_preds + [majority_vote_preds_mean[-1]]
    return majority_vote_preds_mean, majority_vote_preds, each_sample_preds




if __name__=="__main__":
    import json
    import os
    from os.path import join, dirname, basename
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="data/Text_Localization/Goodnews/goodnews_test-gpt2-xl-art1000-seg1.json",
                        help='path to goodnews file')
    parser.add_argument('--sentence_head_folder', type=str, default="logs/sentence_head_goodnews", help="prediction head for sentences within a window")
    parser.add_argument('--cache_dir', type=str, default="/projectnb/ivc-ml/zpzhang/checkpoints/transformers_cache")
    parser.add_argument('--window_size', type=int, default=3)
    parser.add_argument('--article_num', type=int, default=1000) # 2 for debug
    parser.add_argument('--save_folder', type=str, default="results/ablation_study")
    parser.add_argument('--save_name', type=str, default="")
    args = parser.parse_args()
    print(args)


    if args.data_path.split(".")[-1]=="jsonl":
        with open(args.data_path, 'r') as f:
            data = []
            for l_no, l in enumerate(f):
                data.append(json.loads(l))
    elif args.data_path.split(".")[-1]=='json':
        with open(args.data_path,'r') as f:
            data = json.load(f)


    sentence_head_model = torch.load(os.path.join(args.sentence_head_folder, "epoch-best.pkl"), map_location=DEVICE)
    results = run_supervised_experiment_sentence_head(data=data[0: args.article_num], cache_dir=args.cache_dir,
                              DEVICE=DEVICE, pos_bit=0, num_labels=2, window_size=args.window_size, sentence_head_model=sentence_head_model)
    print("AP results: ", results['AP'])
    print("dataset_AP results: ", results['dataset_AP'])
    # print(results['AP_list'])


    AP_results = {
        "AP": results['AP'],
        "dataset_AP": results['dataset_AP'],
        # "dataset_singleAP": results['dataset_singleAP'],
        "args": vars(args),
        "AP_list": results['AP_list'],
        "ground_truth": results["ground_truth"],
        "prediction": results["prediction"],
    }

    os.makedirs(args.save_folder,exist_ok=True)
    with open(join(args.save_folder, args.save_name+".json"),"w") as f:
        json.dump(AP_results,f,indent=2, default=str)






















