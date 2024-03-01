# Machine-generated Text Localization


**Machine-generated Text Localization** is a task aiming at recognizing machine-generated sentences within a document.

If you find this code useful in your research, please consider citing for our [paper](https://arxiv.org/pdf/2402.11744.pdf). 

    @article{zhang2024machinegenerated,
             title={Machine-generated Text Localization},
             author={Zhongping Zhang and Wenda Qin and Bryan A. Plummer},
             journal={arXiv preprint arXiv:2402.11744},
             year={2024}
             }

<!--<img src="figure_overview.png" alt="alt text" style="zoom:50%;" />-->

<div style="text-align: center;">
<img src="figure_overview.png" alt="alt text" width="500" height="400" >
</div>

## TODO LIST
- [x] Gradio apps for Machine-generated Text Localization [[1]](#mgtl) (MGTL)
- [x] Adapt Roberta Models (OpenAI-D [[2]](#openai_d), ChatGPT-D[[3]](#chatgpt_d)) to MGTL
- [x] Adapt Fast-DetectGPT[[4]](#fast_detectgpt) to MGTL
- [ ] Release code for data generation
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

## MGTL by Roberta Detectors
Apply OpenAI-D[[2]](#openai_d) to MGTL
```python
python gradio_MGTL_roberta.py
```
![img](github_figures/screenshot_black_roberta_mgtl.png)

Apply ChatGPT-D[[3]](#chatgpt_d) to MGTL
```python
python gradio_MGTL_roberta.py --model_name=Hello-SimpleAI/chatgpt-detector-roberta
```

## MGTL by Fast-DetectGPT
Apply Fast-DetectGPT[[4]](#fast_detectgpt) to MGTL, where we borrow the implementation code from [their official repo](https://github.com/baoguangsheng/fast-detect-gpt).
```python
python gradio_MGTL_fastdetectgpt.py
```
![img](github_figures/screenshot_black_fastdetectgpt_mgtl.png)
Though DetectGPT[[5]](#detectgpt) series methods are zero-shot methods, they still need training data to determine 
the thresholds. Thus, if you would like to get a decent results on your own data, specific data distribution 
files (*e.g.*, files under [gradio_utils/local_infer_ref](gradio_utils/local_infer_ref)) would be useful.



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


