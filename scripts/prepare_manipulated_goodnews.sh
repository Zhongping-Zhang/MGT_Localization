#conda activate alpaca
export PYTHONPATH=$PWD

GOODNEWS_TRAIN_FILE=data/Goodnews/goodnews_train.jsonl
GOODNEWS_VAL_FILE=data/Goodnews/goodnews_val.jsonl
GOODNEWS_TEST_FILE=data/Goodnews/goodnews_test.jsonl
OUTPUT_FOLDER=data/Text_Localization/Goodnews

train_article_num=10000
test_article_num=1000
model_list=(gpt2-xl EleutherAI/gpt-neo-2.7B EleutherAI/gpt-j-6B facebook/opt-2.7b EleutherAI/gpt-neox-20b)
seg_list=(3 2 1)

#for model_name in ${model_list[@]}; do
#  echo ${model_name}
#  for seg_num in ${seg_list[@]}; do
#    echo segment number ${seg_num}
#    python dataloaders/ArticleBench/manipulate_goodnews_articles.py\
#      --model_name=${model_name}\
#      --file_path=${GOODNEWS_TRAIN_FILE}\
#      --output_dir=${OUTPUT_FOLDER}\
#      --segment_num=${seg_num}\
#      --article_num=${train_article_num}
#  done
#done

for model_name in ${model_list[@]}; do
  echo ${model_name}
  for seg_num in ${seg_list[@]}; do
    echo segment number ${seg_num}
    python dataloaders/ArticleBench/manipulate_goodnews_articles.py\
      --model_name=${model_name}\
      --file_path=${GOODNEWS_TEST_FILE}\
      --output_dir=${OUTPUT_FOLDER}\
      --segment_num=${seg_num} \
      --article_num=${test_article_num}
  done
done





