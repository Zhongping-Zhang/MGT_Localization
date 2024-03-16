import gradio as gr
import argparse
import torch
import transformers
import spacy
nlp = spacy.load("en_core_web_sm")
from gradio_utils import get_binoculars_model_prediction, highlight_text, DEVICE, Binoculars




parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="Binoculars (https://github.com/ahans30/Binoculars)", help="detector name")
parser.add_argument('--cache_dir', type=str, default="/projectnb/ivc-ml/zpzhang/checkpoints/transformers_cache")
args = parser.parse_args()
print(args)

print(f'Beginning Binoculars (https://github.com/ahans30/Binoculars) evaluation with {args.model_name}...')
bino = Binoculars(cache_dir=args.cache_dir)


def article_analysis(input_article, window_size=1, threshold_low=0.50, threshold_high=0.60, strategy="vote"):
    doc = nlp(input_article)
    sentence_spans = list(doc.sents)
    sentence_list = [str(ele) for ele in sentence_spans]

    binoculars_threshold = bino.threshold
    test_preds, test_preds_org, whole_document_score, binoculars_preds, whole_document_binoculars_score, whole_document_label = \
        get_binoculars_model_prediction(sentence_list=sentence_list,
                                        window_size=window_size,
                                        strategy=strategy,
                                        cache_dir=args.cache_dir,
                                        bino=bino)

    test_preds_dict = {"pretrained detector": args.model_name,
                       # "whole document score, machine-generated": "{:.2f}%".format(whole_document_score * 100),
                       # "sentence scores": test_preds,
                       "whole document label": whole_document_label,
                       "whole document Binoculars score": "{:f}".format(whole_document_binoculars_score),
                       "Binoculars Threshold": "{:f}".format(binoculars_threshold),
                       "sentence Binoculars scores": binoculars_preds,}
    token_highlight = highlight_text(sentence_list, test_preds, threshold_low=threshold_low,
                                     threshold_high=threshold_high)

    return token_highlight, test_preds_dict


example2=['''Dr. Capy Cosmos, a capybara unlike any other, astounded the scientific community with his \
groundbreaking research in astrophysics. With his keen sense of observation and unparalleled ability to interpret \
cosmic data, he uncovered new insights into the mysteries of black holes and the origins of the universe. As he \
peered through telescopes with his large, round eyes, fellow researchers often remarked that it seemed as if the \
stars themselves whispered their secrets directly to him. Dr. Cosmos not only became a beacon of inspiration to \
aspiring scientists but also proved that intellect and innovation can be found in the most unexpected of creatures.''']

example1=['''Machine-Generated Text (MGT) detection aims to identify a piece of text as machine or human written. \
Prior work has primarily formulated MGT as a binary classification task over an entire document, with limited work \
exploring cases where only part of a document is machine generated. This paper provides the first in-depth study \
of MGT that localizes the portions of a document that were machine generated. Thus, if a bad actor were to change \
a key portion of a news article to spread misinformation, whole document MGT detection may fail since the vast majority \
is human written, but our approach can succeed due to its granular approach. A key challenge in our MGT localization \
task is that short spans of text, e.g., a single sentence, provides little information indicating if it is machine \
generated due to its short length. To address this, we leverage contextual information, where we predict whether \
multiple sentences are machine or human written at once.  This enables our approach to identify changes in style \
or content to boost performance. A gain of 4-13% mean Average Precision (mAP) over prior work demonstrates the \
effectiveness of approach on five diverse datasets: GoodNews, VisualNews, WikiText, Essay, and WP.''']


example3=['''Maria C. Carrillo, vice president of medical and scientific relations at the Alzheimer's Association, \
said the results would come quickly. Within a few years, as researchers simultaneously compare the three approaches \
to stopping the disease, they should know which drug, if any, is going to work.  Carrillo said. "If there is a drug \
that works, we are going to be the ones to take it and test it," she said, "We are not going to be the ones to say no, \
But what about the people whose lives are most at risk?"  The announcement comes at a time of transition for Alzheimer's \
research. …… The drugs were chosen from among 15 that drug companies offered, said the study's principal investigator, \
Dr. Randall Bateman of the Washington University School of Medicine in St. Louis. Shouldn't a drug in development get \
tested in people who will be the most affected? The answer is no. The studies were not designed to test drugs in people \
who are at the highest risk for Alzheimer's disease. Because of that, their findings could have huge consequences for \
those in other developing countries.  One concern is something called ARIA, for amyloid related imaging abnormality. \
People with the abnormality may have no signs that anything is wrong, but brain scans show what looks like a change in \
neural connections. …… Researchers said they would face that issue when they come to it. “The study in the U.S., our \
conclusion is that we can't be confident in saying these drugs will work in the vast majority of the population,” said \
Dr. William M. Foege, an associate professor of neurology and psychiatry at the University of California, San Francisco, \
"The study also showed that some of the drugs were unlikely to save lives. For example, the drug metformin, which can \
raise blood sugar, has so much side effects that most people with diabetes are put off by its side effects and don't \
use it at all.” Then we can put pressure on to bring down the cost."''']




block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Machine-generated Text Localization (Binoculars)")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Input Article: ")
            strategy = gr.Radio(["single-sentence", "multi-sentence"], label="strategy for MGT detectors", value="multi-sentence")
            run_button = gr.Button(value="Run") # label="Run" for older Gradio version
            with gr.Accordion("Advanced options for Binoculars", open=False):
                window_size = gr.Slider(label="number of sentences per window", minimum=1, maximum=10, value=3, step=1)
                threshold_low = gr.Slider(label="lower threshold for MGT", minimum=0, maximum=1, value=0.5, step=0.01)
                threshold_high = gr.Slider(label="upper threshold for MGT", minimum=0, maximum=1, value=0.6, step=0.01)


            examples = gr.Examples(examples=[example2, example1, example3],
                                   inputs=[prompt],
                                   )

        with gr.Column():
            result_highlight = gr.HighlightedText(label="highlight machine-generated text", show_label=True,
                combine_adjacent=True,
                show_legend=True,
                color_map={"human": "green", "likely-machine": "yellow", "machine": "red"})
            result_json = gr.Json(label="MGTL analysis json results", show_label=True,)
            result_html = gr.HTML(label="MGTL HTML visualization", show_label=True)

    run_button.click(fn=article_analysis, inputs=[prompt, window_size, threshold_low, threshold_high, strategy], outputs=[result_highlight, result_json])



block.launch(server_name='0.0.0.0', share=True)




