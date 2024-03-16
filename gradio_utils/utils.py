import torch
import transformers

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROBERTA_MAX_TEXT_LENGTH=512

def highlight_text(sentence_list, prediction_score_list, threshold_low=0.50, threshold_high=0.60):
    """
    highlight human-written text by green, machine-generated text by red, likely-machine-generated text by yellow
    """
    token_list = []
    for idx in range(len(sentence_list)):
        sentence = sentence_list[idx]
        prediction_score = prediction_score_list[idx]
        for token in sentence:
            if 0<=prediction_score<threshold_low:
                token_label="human"
            elif threshold_low<=prediction_score<threshold_high:
                token_label="likely-machine"
            elif threshold_high<prediction_score:
                token_label="machine"
            else:
                assert False, "prediction score out of distribution"
            token_list.append((token, token_label))
    return token_list



def get_supervised_model_prediction(model, tokenizer, sentence_list, pos_bit=0, window_size=1, window_step=1, strategy="multi-sentence"):
    """ prediction labels for the input sentence list, 1 corresponds to Machine-generated Text,
    this function can be used for model-based methods such as OpenAI-Detector or ChatGPT-Detector.
    :param model:
    :param tokenizer:
    :param sentence_list:
    :param pos_bit:
    :param window_size: model becomes more reliable after around 50 tokens
    :param window_step:
    :return:
    """
    if strategy=="single-sentence":
        window_size=1

    with torch.no_grad():
        majority_vote_preds = [[] for i in range(len(sentence_list))]

        # calculate prediction score for the whole document
        text_merge = " ".join(sentence_list)
        window_token_data = tokenizer(text_merge, padding=True, truncation=True,
                                      max_length=ROBERTA_MAX_TEXT_LENGTH, return_tensors="pt").to(DEVICE)
        whole_document_score = model(**window_token_data).logits.softmax(-1)[:, pos_bit].tolist()[0]
        # majority_vote_preds = [[prediction_score] for i in range(len(sentence_list))]

        for window_start in range(0, max(1, len(sentence_list)-window_size+1), window_step):
            text_data = sentence_list[window_start:window_start+window_size]
            text_merge = " ".join(text_data)
            window_token_data = tokenizer(text_merge, padding=True, truncation=True,
                                          max_length=ROBERTA_MAX_TEXT_LENGTH, return_tensors="pt").to(DEVICE)
            prediction_score = model(**window_token_data).logits.softmax(-1)[:, pos_bit].tolist()[0]

            try:
                for vote_idx in range(window_start, window_start+window_size):
                    majority_vote_preds[vote_idx].append(prediction_score)
            except:
                for vote_idx in range(window_start, min(window_start + window_size, len(sentence_list))):
                    majority_vote_preds[vote_idx].append(prediction_score)

    majority_vote_preds_mean = [(sum(sub_list)/len(sub_list)) for sub_list in majority_vote_preds]
    return majority_vote_preds_mean, majority_vote_preds, whole_document_score


# functions for Fast-DetectGPT (https://github.com/baoguangsheng/fast-detect-gpt)
from gradio_utils.fastdetectgpt_scripts.model import load_tokenizer, load_model
from gradio_utils.fastdetectgpt_scripts.utils_fastdetectgpt import fastdetectgpt_args, get_fastdetectgpt_score
def get_metric_model_prediction(args, sentence_list, window_size=1, window_step=1, strategy="multi-sentence"):
    """ prediction labels for the input sentence list, 1 corresponds to Machine-generated Text,
    this function can be used for metric-based methods such as DetectGPT, Fast-DetectGPT, DetectLLM.
    :param args:
    :param sentence_list:
    :param window_size: model becomes more reliable after around 50 tokens
    :param window_step:
    :return:
    """
    if strategy=="single-sentence":
        window_size=1

    # load model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.dataset, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval()

    with torch.no_grad():
        majority_vote_preds = [[] for i in range(len(sentence_list))]

        # calculate prediction score for the whole document
        text_merge = " ".join(sentence_list)
        crit, whole_document_score = get_fastdetectgpt_score(args, input_text = text_merge,
            scoring_tokenizer=scoring_tokenizer,scoring_model=scoring_model)

        for window_start in range(0, max(1, len(sentence_list)-window_size+1), window_step):
            text_data = sentence_list[window_start:window_start+window_size]
            text_merge = " ".join(text_data)
            crit, prediction_score = get_fastdetectgpt_score(args, input_text = text_merge,
                scoring_tokenizer=scoring_tokenizer,scoring_model=scoring_model)

            try:
                for vote_idx in range(window_start, window_start+window_size):
                    majority_vote_preds[vote_idx].append(prediction_score)
            except:
                for vote_idx in range(window_start, min(window_start + window_size, len(sentence_list))):
                    majority_vote_preds[vote_idx].append(prediction_score)

    majority_vote_preds_mean = [(sum(sub_list)/len(sub_list)) for sub_list in majority_vote_preds]
    return majority_vote_preds_mean, majority_vote_preds, whole_document_score



# functions for Binocular (https://github.com/ahans30/Binoculars)
# functions for Binocular (https://github.com/ahans30/Binoculars)
from gradio_utils.binoculars import Binoculars

def get_binoculars_model_prediction(sentence_list, window_size=1, window_step=1, strategy="multi-sentence", cache_dir=".cache", bino=None,
                                    return_binary_label=True,):
    """ prediction labels for the input sentence list, 1 corresponds to Machine-generated Text,
    this function is used for Binocular
    :param args:
    :param sentence_list:
    :param window_size: model becomes more reliable after around 50 tokens
    :param window_step:
    :return:
    """
    if strategy=="single-sentence":
        window_size=1

    # load model
    if bino is None:
        bino = Binoculars(cache_dir=cache_dir)
    binoculars_threshold = bino.threshold

    with torch.no_grad():
        majority_vote_preds = [[] for i in range(len(sentence_list))]
        binoculars_vote_preds = [[] for i in range(len(sentence_list))]

        # calculate prediction score for the whole document
        text_merge = " ".join(sentence_list)
        whole_document_binoculars_score = bino.compute_score(text_merge)
        whole_document_score = -(whole_document_binoculars_score-binoculars_threshold)+0.5  # when Binoculars score< threshold, likely AI-generated; Otherwise, human-generated
        whole_document_score = min(max(whole_document_score, 0), 1)
        whole_document_label = bino.predict(text_merge)

        for window_start in range(0, max(1, len(sentence_list)-window_size+1), window_step):
            text_data = sentence_list[window_start:window_start+window_size]
            text_merge = " ".join(text_data)
            binoculars_score = bino.compute_score(text_merge)
            prediction_score = -(binoculars_score-binoculars_threshold)+0.5
            prediction_score = min(max(prediction_score, 0), 1)

            try:
                for vote_idx in range(window_start, window_start+window_size):
                    majority_vote_preds[vote_idx].append(prediction_score)
                    binoculars_vote_preds[vote_idx].append(binoculars_score)
            except:
                for vote_idx in range(window_start, min(window_start + window_size, len(sentence_list))):
                    majority_vote_preds[vote_idx].append(prediction_score)
                    binoculars_vote_preds[vote_idx].append(binoculars_score)

    majority_vote_preds_mean = [(sum(sub_list)/len(sub_list)) for sub_list in majority_vote_preds]
    binoculars_vote_preds_mean = [(sum(sub_list) / len(sub_list)) for sub_list in binoculars_vote_preds]

    return majority_vote_preds_mean, majority_vote_preds, whole_document_score, binoculars_vote_preds_mean, whole_document_binoculars_score, whole_document_label



