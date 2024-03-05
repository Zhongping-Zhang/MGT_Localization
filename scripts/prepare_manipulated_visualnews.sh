#conda activate alpaca
export PYTHONPATH=$PWD

VISUALNEWS_TRAIN_FILE=data/Visualnews/filtereddata/visualnews_train.jsonl
VISUALNEWS_VAL_FILE=data/Visualnews/filtereddata/visualnews_val.jsonl
VISUALNEWS_TEST_FILE=data/Visualnews/filtereddata/visualnews_test.jsonl
OUTPUT_FOLDER=data/Text_Localization/Visualnews

train_article_num=10000
test_article_num=1100
model_list=(gpt2-xl EleutherAI/gpt-neo-2.7B EleutherAI/gpt-j-6B facebook/opt-2.7b EleutherAI/gpt-neox-20b)
seg_list=(3 2 1)


for model_name in ${model_list[@]}; do
  echo ${model_name}
  for seg_num in ${seg_list[@]}; do
    python dataloaders/ArticleBench/manipulate_visualnews_articles.py\
      --model_name=${model_name}\
      --file_path=${VISUALNEWS_TEST_FILE}\
      --output_dir=${OUTPUT_FOLDER}\
      --segment_num=${seg_num} \
      --article_num=${test_article_num}
  done
done



#for model_name in ${model_list[@]}; do
#  echo ${model_name}
#  for seg_num in ${seg_list[@]}; do
#    python dataloaders/ArticleBench/manipulate_visualnews_articles.py\
#      --model_name=${model_name}\
#      --file_path=${VISUALNEWS_TRAIN_FILE}\
#      --output_dir=${OUTPUT_FOLDER}\
#      --segment_num=${seg_num} \
#      --article_num=${train_article_num}
#  done
#done

