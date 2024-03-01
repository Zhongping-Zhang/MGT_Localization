import gradio as gr
import argparse
import torch
import transformers
import spacy
# import os
# os.system('python -m spacy download en_core_web_sm') # uncomment this if en_core_web_sm is not installed
nlp = spacy.load("en_core_web_sm")
from gradio_utils import get_supervised_model_prediction, highlight_text, DEVICE




parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="roberta-large-openai-detector", help="supervised detector name")
parser.add_argument('--cache_dir', type=str, default=".cache")
args = parser.parse_args()
print(args)

print(f'Beginning supervised evaluation with {args.model_name}...')


POS_BIT = 0 if "openai" in args.model_name else 1
PRETRAINED_DETECTOR_LIST=["roberta-large-openai-detector","roberta-base-openai-detector",
                          "Hello-SimpleAI/chatgpt-detector-roberta"]
if args.model_name in PRETRAINED_DETECTOR_LIST:
    DETECTOR = transformers.AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=args.model_name, num_labels=2, cache_dir=args.cache_dir).to(DEVICE)
    TOKENIZER = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model_name, cache_dir=args.cache_dir)


def article_analysis(input_article, window_size=1, pos_bit=1, threshold_low=0.50, threshold_high=0.60,
                     strategy="vote"):
    doc = nlp(input_article)
    sentence_spans = list(doc.sents)
    sentence_list = [str(ele) for ele in sentence_spans]

    test_preds, test_preds_org, whole_document_score = get_supervised_model_prediction(model=DETECTOR,
                                                                                       tokenizer=TOKENIZER,
                                                                                       sentence_list=sentence_list,
                                                                                       pos_bit=pos_bit,
                                                                                       window_size=window_size,
                                                                                       strategy=strategy)

    test_preds_dict = {"pretrained detector": args.model_name,
                       "whole document score, machine-generated": "{:.2f}%".format(whole_document_score * 100),
                       "sentence scores": test_preds}
    token_highlight = highlight_text(sentence_list, test_preds, threshold_low=threshold_low,
                                     threshold_high=threshold_high)

    return token_highlight, test_preds_dict


example1=["Maria C. Carrillo, vice president of medical and scientific relations at the Alzheimer's Association, said the results would come quickly. Within a few years, as researchers simultaneously compare the three approaches to stopping the disease, they should know which drug, if any, is going to work.  Carrillo said. “If there is a drug that works, we are going to be the ones to take it and test it,” she said, “We are not going to be the ones to say no, But what about the people whose lives are most at risk?”  The announcement comes at a time of transition for Alzheimer's research. …… The drugs were chosen from among 15 that drug companies offered, said the study's principal investigator, Dr. Randall Bateman of the Washington University School of Medicine in St. Louis. Shouldn't a drug in development get tested in people who will be the most affected? The answer is no. The studies were not designed to test drugs in people who are at the highest risk for Alzheimer's disease. Because of that, their findings could have huge consequences for those in other developing countries.  One concern is something called ARIA, for amyloid related imaging abnormality. People with the abnormality may have no signs that anything is wrong, but brain scans show what looks like a change in neural connections. …… "]
          # "Researchers said they would face that issue when they come to it. “The study in the U.S., our conclusion is that we can't be confident in saying these drugs will work in the vast majority of the population,” said Dr. William M. Foege, an associate professor of neurology and psychiatry at the University of California, San Francisco, “The study also showed that some of the drugs were unlikely to save lives. For example, the drug metformin, which can raise blood sugar, has so much side effects that most people with diabetes are put off by its side effects and don't use it at all.” Then we can put pressure on to bring down the cost."]
example2=["The result was that the United States was eliminated in the semifinals of the competition it had waited so long for and that its captain, Carlos Bocanegra, was yelled at repeatedly by the U.S. Soccer captain Carlos Bocanegra. The final result gave the Americans a chance to regain their title but was far from the perfection fans had been hoping for.\n\nThen in early September came the Washington Nationals' first walk-off victory in the World Series since 2006."]



block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Machine-generated Text Localization (Roberta Detector)")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Input Article: ")
            strategy = gr.Radio(["single-sentence", "multi-sentence"], label="strategy for MGT detectors", value="multi-sentence")
            run_button = gr.Button(value="Run") # label="Run" for older Gradio version
            with gr.Accordion("Advanced options", open=False):
                window_size = gr.Slider(label="number of sentences per window", minimum=1, maximum=10, value=3, step=1)
                pos_bit = gr.Slider(label="pos_bit for detector (OpenAI-D=0; ChatGPT-D=1)", minimum=0, maximum=1, value=POS_BIT, step=1)
                threshold_low = gr.Slider(label="lower threshold for MGT", minimum=0, maximum=1, value=0.5, step=0.01)
                threshold_high = gr.Slider(label="upper threshold for MGT", minimum=0, maximum=1, value=0.6, step=0.01)

            examples = gr.Examples(examples=[example1, example2],
                                   inputs=[prompt],
                                   )

        with gr.Column():
            result_highlight = gr.HighlightedText(label="highlight machine-generated text", show_label=True,
                combine_adjacent=True,
                show_legend=True,
                color_map={"human": "green", "likely-machine": "yellow", "machine": "red"})
            result_json = gr.Json(label="MGTL analysis json results", show_label=True,)
            result_html = gr.HTML(label="MGTL HTML visualization", show_label=True)

    run_button.click(fn=article_analysis, inputs=[prompt, window_size, pos_bit, threshold_low, threshold_high, strategy], outputs=[result_highlight, result_json])


block.launch(server_name='0.0.0.0', share=True)
