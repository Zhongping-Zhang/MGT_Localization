#conda activate mgtl

DATA_FOLDER=data/Text_Localization/Wikitext
SENTENCE_HEAD_FOLDER=logs/adaloc_goodnews
SAVE_FOLDER=results/adaloc/wikitext2_log-goodnews

export PYTHONPATH=$PWD

LLM_list=(gpt2-xl EleutherAI_gpt-j-6B EleutherAI_gpt-neo-2.7B EleutherAI_gpt-neox-20b facebook_opt-2.7b)

for LLM_name in ${LLM_list[@]}; do
  echo ${LLM_name}
  python AdaLoc/inference.py\
    --data_path=${DATA_FOLDER}/wikitext2_test-${LLM_name}-art60-seg1.json\
    --sentence_head_folder=${SENTENCE_HEAD_FOLDER}\
    --article_num=1000\
    --save_folder=${SAVE_FOLDER}\
    --save_name=${LLM_name}_sentence_head_seg1

  python AdaLoc/inference.py\
    --data_path=${DATA_FOLDER}/wikitext2_test-${LLM_name}-art60-seg2.json\
    --sentence_head_folder=${SENTENCE_HEAD_FOLDER}\
    --article_num=1000\
    --save_folder=${SAVE_FOLDER}\
    --save_name=${LLM_name}_sentence_head_seg2

  python AdaLoc/inference.py\
    --data_path=${DATA_FOLDER}/wikitext2_test-${LLM_name}-art60-seg3.json\
    --sentence_head_folder=${SENTENCE_HEAD_FOLDER}\
    --article_num=1000\
    --save_folder=${SAVE_FOLDER}\
    --save_name=${LLM_name}_sentence_head_seg3
done









