# Machine-generated Text Localization


**Machine-generated Text Localization** is a task aiming at recognizing machine-generated sentences within a document.
![img](github_figures/MGTL_demo.gif)

If you find this code useful in your research, please consider citing our [paper](https://arxiv.org/pdf/2402.11744.pdf):

    @article{zhang2024machinegenerated,
             title={Machine-generated Text Localization},
             author={Zhongping Zhang and Wenda Qin and Bryan A. Plummer},
             journal={arXiv preprint arXiv:2402.11744},
             year={2024}
             }


<!--<div style="text-align: center;">
<img src="figure_overview.png" alt="alt text" width="500" height="400" >
</div>-->




## TODO LIST
- [x] Gradio apps for Machine-generated Text Localization [[1]](#mgtl) (MGTL)
- [x] Apply Roberta Models (OpenAI-D [[2]](#openai_d), ChatGPT-D[[3]](#chatgpt_d)) to MGTL
- [x] Apply Fast-DetectGPT[[4]](#fast_detectgpt) to MGTL
- [x] Release code for data generation
- [ ] Release code for AdaLoc


## Setting up the Environment
We provide two options to create an environment for MGTL. You can either create a new conda environment
```shell
conda env create -f environment.yml
conda activate mgtl
```
or set up the environment by pip
```shell
pip install -r requirements.txt
```

If spaCy is not installed before in your machine, the following command might be useful 
```shell
python -m spacy download en_core_web_sm
```

# Interactive Apps for MGTL
In this section, we provide interactive apps for the MGTL task. We have integrated OpenAI-Detector [[2]](#openai_d), 
ChatGPT-Detector [[3]](#chatgpt_d), and Fast-DetectGPT [[4]](#fast_detectgpt) into our interactive platform as examples.
Feel free to plug in your preferred/developed method!

## MGTL by Roberta Detectors
Integrate OpenAI-Detector to MGTL
```python
python gradio_MGTL_roberta.py
```
![img](github_figures/screenshot_black_roberta_mgtl.png)

Integrate ChatGPT-Detector to MGTL
```python
python gradio_MGTL_roberta.py --model_name=Hello-SimpleAI/chatgpt-detector-roberta
```

## MGTL by Fast-DetectGPT
Integrate Fast-DetectGPT to MGTL. We borrow the implementation code from [their official repo](https://github.com/baoguangsheng/fast-detect-gpt).
```python
python gradio_MGTL_fastdetectgpt.py
```
![img](github_figures/screenshot_black_fastdetectgpt_mgtl.png)
Though DetectGPT[[5]](#detectgpt) series methods are zero-shot methods, they still need training data to determine 
the thresholds. Otherwise, methods like Fast-DetectGPT can predict most-likely machine-generated sentences within an article, while cannot 
accurately determine whether these sentences are machine-generated. Thus, if you would like to get a decent results on 
your own data, specific data distribution files (*e.g.*, files under [gradio_utils/local_infer_ref](gradio_utils/local_infer_ref)) 
would be useful.

## Data Preparation

Since Essay and WP datasets already provide machine-generated text, we directly mix them using our *merge_sentences* 
function in [dataloader_utils.py](dataloaders/dataloader_utils.py). For GoodNews, VisualNews, and WikiText, run the 
following scripts to insert machine-generated sentences into human-written articles.    
```sh
sh scripts/prepare_manipulated_goodnews.sh
sh scripts/prepare_manipulated_visualnews.sh
sh scripts/prepare_manipulated_wikitext.sh
```
We provide the original articles of these datasets on [Google Drive](https://drive.google.com/drive/folders/1KmtlbHlwp2piuZIKx-HVKO3N2dRAQFjY?usp=sharing).



## Acknowledgement
We appreciate the following projects (and many other open source projects not listed here): 

[GPT2-Detector](https://openai-openai-detector.hf.space) &#8194;
[ChatGPT-Detector](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection) &#8194; 
[DetectGPT](https://github.com/eric-mitchell/detect-gpt) &#8194; 
[FastDetectGPT](https://github.com/baoguangsheng/fast-detect-gpt) &#8194; 
[GhostBuster](https://github.com/vivek3141/ghostbuster) &#8194;
[MGTBench](https://github.com/xinleihe/MGTBench) &#8194;


## Reference 
<a id="mgtl">[1]</a>
Zhang, Zhongping, Wenda Qin, and Bryan A. Plummer. "Machine-generated Text Localization." arXiv 2024. 

<a id="openai_d">[2]</a>
Solaiman, Irene, et al. "Release strategies and the social impacts of language models." arXiv 2019.

<a id="chatgpt_d">[3]</a>
Guo, Biyang, et al. "How close is chatgpt to human experts? comparison corpus, evaluation, and detection." arXiv 2023.

<a id="fast_detectgpt">[4]</a>
Bao, Guangsheng, et al. "Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature." ICLR 2023.

<a id="detectgpt">[5]</a>
Mitchell, Eric, et al. "DetectGPT: Zero-Shot Machine-Generated Text Detection Using Probability Curvature" ICML 2023.


