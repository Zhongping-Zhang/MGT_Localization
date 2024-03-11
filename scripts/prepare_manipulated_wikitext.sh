#conda activate mgtl
export PYTHONPATH=$PWD

WIKITEXT_TEST_FILE=data/WikiText/wikitext2_test.jsonl
OUTPUT_FOLDER=data/Text_Localization/Wikitext


test_article_num=60
model_list=(gpt2-xl EleutherAI/gpt-neo-2.7B EleutherAI/gpt-j-6B facebook/opt-2.7b EleutherAI/gpt-neox-20b)
seg_list=(3 2 1)
#model_list=(EleutherAI/gpt-neox-20b)


for model_name in ${model_list[@]}; do
  echo ${model_name}
  for seg_num in ${seg_list[@]}; do
    python dataloaders/ArticleBench/manipulate_wikitext_articles.py\
      --model_name=${model_name}\
      --file_path=${WIKITEXT_TEST_FILE}\
      --output_dir=${OUTPUT_FOLDER}\
      --segment_num=${seg_num}\
      --article_num=${test_article_num}\
      --random_seed=$((${seg_num}+345))
  done
done





