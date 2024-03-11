# conda activate mgtl

export PYTHONPATH=$PWD

DATA_FOLDER=data/Text_Localization/Goodnews


python AdaLoc/train.py \
  --model_name=adaloc_goodnews\
  --train_file=${DATA_FOLDER}/goodnews_train-gpt2-xl-art10000-seg2.json\
  --test_file=${DATA_FOLDER}/goodnews_val-gpt2-xl-art1000-seg3.json\
  --roberta_detector_name=roberta-large-openai-detector\
  --num_epoch=3\
  --batch_size=16





