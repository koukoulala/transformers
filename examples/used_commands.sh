# 8.9
python examples/pytorch/summarization/run_summarization.py  --model_name_or_path t5-small  --do_train  --do_eval  --dataset_name cnn_dailymail  --dataset_config "3.0.0"  --source_prefix "summarize: "  --output_dir ./results/t5_cnn_dailymail --per_device_train_batch_size=4 --per_device_eval_batch_size=4 --overwrite_output_dir --predict_with_generate
nohup python -u examples/pytorch/summarization/run_xglue.py  --model_name_or_path 'microsoft/xprophetnet-large-wiki100-cased-xglue-ntg'  --do_eval  --do_predict --dataset_name xglue  --dataset_config ntg   --output_dir ./results/xlm_pro_ntg_test --per_device_train_batch_size=4 --per_device_eval_batch_size=4 --overwrite_output_dir --predict_with_generate --cache_dir=../ckpt/xprophtnet_ntg --data_cache_dir=../datasets/transformer_ntg &> logs/xlm_pro_ntg_test.out &

#8.11
nohup python -u examples/pytorch/summarization/run_xglue_no_trainer.py  --model_name_or_path 'microsoft/xprophetnet-large-wiki100-cased-xglue-ntg'  --dataset_name xglue  --dataset_config_name ntg   --output_dir ./results/xpro_ntg --per_device_train_batch_size=4 --per_device_eval_batch_size=4 --cache_dir=../ckpt/xprophtnet_ntg --data_cache_dir=../datasets/transformer_ntg &> logs/xpro_ntg.out &
nohup python -u examples/pytorch/summarization/run_xglue_no_trainer.py  --model_name_or_path 'microsoft/xprophetnet-large-wiki100-cased-xglue-ntg'  --dataset_name xglue  --dataset_config_name ntg   --output_dir ./results/xpro_ntg --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --cache_dir=../ckpt/xprophtnet_ntg --data_cache_dir=../datasets/transformer_ntg --max_source_length=128 &> logs/xpro_ntg.out &
python examples/pytorch/summarization/preprocess_data.py
CUDA_VISIBLE_DEVICES=1,0 nohup python -u examples/pytorch/summarization/run_xglue_no_trainer.py  --model_name_or_path 'microsoft/xprophetnet-large-wiki100-cased-xglue-ntg'  --dataset_name ntg --data_folder=../datasets/xglue_full_dataset/sampled_NTG  --output_dir=./results/xpro_ntg --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --cache_dir=../ckpt/xprophtnet_ntg --max_source_length=128 &> logs/xpro_ntg.out &

"pip install --user . && pip install --user numpy==1.20.0 && pip install -r ./examples/pytorch/summarization/requirements.txt && python --version && python -u examples/pytorch/summarization/run_xglue_no_trainer.py --model_name_or_path 'microsoft/xprophetnet-large-wiki100-cased-xglue-ntg'  --dataset_name=ntg  --data_folder=[#input-training-data-path] --output_dir=./results/xpro_ntg --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --cache_dir=[#input-previous-model-path]"

#8.12
CUDA_VISIBLE_DEVICES=1,0 nohup python -u examples/pytorch/summarization/run_xglue_no_trainer.py  --model_name_or_path 'microsoft/xprophetnet-large-wiki100-cased-xglue-ntg'  --dataset_name=ntg --data_folder=../datasets/xglue_full_dataset/sampled_NTG  --output_dir=./results/xpro_ntg --per_device_train_batch_size=1 --per_device_eval_batch_size=2 --cache_dir=../ckpt/xprophtnet_ntg --max_source_length=128 &> logs/xpro_ntg.out &
nohup python -u examples/pytorch/summarization/preprocess_data.py &> logs/preprocess_data.out &
CUDA_VISIBLE_DEVICES=0 nohup python -u examples/pytorch/summarization/run_xglue_no_trainer.py  --model_name_or_path 'microsoft/xprophetnet-large-wiki100-cased-xglue-ntg'  --dataset_name=ntg --data_folder=../datasets/xglue_full_dataset/sampled_NTG  --output_dir=./results/xpro_ntg --per_device_train_batch_size=1 --per_device_eval_batch_size=32 --cache_dir=../ckpt/xprophtnet_ntg --overwrite_processed=True &> logs/xpro_ntg_0.out &
CUDA_VISIBLE_DEVICES=0 nohup python -u examples/pytorch/summarization/run_xglue_no_trainer.py  --model_name_or_path 'microsoft/xprophetnet-large-wiki100-cased-xglue-ntg'  --dataset_name=ntg --data_folder=../datasets/xglue_full_dataset/sampled_NTG  --output_dir=./results/xpro_ntg --per_device_train_batch_size=1 --per_device_eval_batch_size=32 --cache_dir=../ckpt/xprophtnet_ntg &> logs/xpro_ntg_0.out &
nohup python -u examples/pytorch/summarization/run_xglue_no_trainer.py  --model_name_or_path 'microsoft/xprophetnet-large-wiki100-cased-xglue-ntg'  --dataset_name=ntg --data_folder=../datasets/xglue_full_dataset/sampled_NTG  --output_dir=./results/xpro_ntg --per_device_train_batch_size=1 --per_device_eval_batch_size=16 --cache_dir=../ckpt/xprophtnet_ntg &> logs/xpro_ntg_1.out &
nohup python -u -m torch.distributed.launch --nproc_per_node 2 --use_env examples/pytorch/summarization/run_xglue_no_trainer.py  --model_name_or_path 'microsoft/xprophetnet-large-wiki100-cased-xglue-ntg'  --dataset_name=ntg --data_folder=../datasets/xglue_full_dataset/sampled_NTG  --output_dir=./results/xpro_ntg --per_device_train_batch_size=1 --per_device_eval_batch_size=2 --cache_dir=../ckpt/xprophtnet_ntg &> logs/xpro_ntg_1.out &
CUDA_VISIBLE_DEVICES=1,0 accelerate launch examples/pytorch/summarization/run_xglue_no_trainer.py  --model_name_or_path 'microsoft/xprophetnet-large-wiki100-cased-xglue-ntg'  --dataset_name=ntg --data_folder=../datasets/xglue_full_dataset/sampled_NTG  --output_dir=./results/xpro_ntg --per_device_train_batch_size=1 --per_device_eval_batch_size=2 --cache_dir=../ckpt/xprophtnet_ntg &> logs/xpro_ntg_2.out &

# 8.13
nohup python -u -m torch.distributed.launch --nproc_per_node 2 --use_env examples/pytorch/summarization/run_xglue_no_trainer.py  --model_name_or_path 'microsoft/xprophetnet-large-wiki100-cased-xglue-ntg'  --dataset_name=ntg --data_folder=../datasets/xglue_full_dataset/sampled_NTG  --output_dir=./results/xpro_ntg --per_device_train_batch_size=1 --per_device_eval_batch_size=2 --cache_dir=../ckpt/xprophtnet_ntg --num_beams=4 &> logs/xpro_ntg_0.out &
nohup accelerate launch examples/pytorch/summarization/run_xglue_no_trainer.py  --model_name_or_path 'microsoft/xprophetnet-large-wiki100-cased-xglue-ntg'  --dataset_name=ntg --data_folder=../datasets/xglue_full_dataset/sampled_NTG  --output_dir=./results/xpro_ntg --per_device_train_batch_size=1 --per_device_eval_batch_size=2 --cache_dir=../ckpt/xprophtnet_ntg --num_beams=4 &> logs/xpro_ntg_1.out &

# 8.16
nohup python -u -m torch.distributed.launch --nproc_per_node 2 --use_env examples/pytorch/summarization/run_xglue_no_trainer.py  --model_name_or_path 'microsoft/xprophetnet-large-wiki100-cased-xglue-ntg'  --dataset_name=ntg --data_folder=../datasets/xglue_full_dataset/sampled_NTG  --output_dir=./results/xpro_ntg --per_device_train_batch_size=1 --per_device_eval_batch_size=2 --cache_dir=../ckpt/xprophtnet_ntg --num_beams=4 --max_source_length=512 --pad_to_max_length --overwrite_processed=True &> logs/xpro_ntg_0.out &
nohup python -u -m torch.distributed.launch --nproc_per_node 2 --use_env examples/pytorch/summarization/run_xglue_no_trainer.py  --model_name_or_path 'microsoft/xprophetnet-large-wiki100-cased-xglue-ntg'  --dataset_name=ntg --data_folder=../datasets/xglue_full_dataset/sampled_NTG  --output_dir=./results/xpro_ntg --per_device_train_batch_size=1 --per_device_eval_batch_size=2 --cache_dir=../ckpt/xprophtnet_ntg --num_beams=10 --max_source_length=512 --pad_to_max_length &> logs/xpro_ntg_0.out &

# 8.17
nohup python -u -m torch.distributed.launch --nproc_per_node 2 --use_env examples/pytorch/summarization/run_xglue_no_trainer.py  --model_name_or_path="google/mt5-large"  --dataset_name=ntg --data_folder=../datasets/xglue_full_dataset/NTG  --output_dir=./results/mT5_large_ntg --per_device_train_batch_size=4 --per_device_eval_batch_size=4 --cache_dir=../ckpt/mT5_large &> logs/mT5_large_ntg_0.out &
"python -m pip install virtualenv --user && python -m virtualenv /tmp/env_mt5 && . /tmp/env_mt5/bin/activate && python -m pip install --editable .  && pip install -r ./examples/pytorch/summarization/requirements.txt && python -m pip install torch==1.5.0 && python --version && python -u -m torch.distributed.launch --nproc_per_node 2 --use_env examples/pytorch/summarization/run_xglue_no_trainer.py --model_name_or_path=google/mt5-large  --dataset_name=ntg  --data_folder=[#input-training-data-path] --output_dir=./results/mT5_large_ntg --per_device_train_batch_size=6 --per_device_eval_batch_size=6 --cache_dir=[#input-previous-model-path]"

# 8.18
"python -m pip install virtualenv --user && python -m virtualenv /tmp/env_mt5 && . /tmp/env_mt5/bin/activate && python -m pip install --editable .  && pip install -r ./examples/pytorch/summarization/requirements.txt && python -m pip install torch==1.5.0 && python --version && python -u -m torch.distributed.launch --nproc_per_node 4 --use_env examples/pytorch/summarization/run_xglue_no_trainer.py --model_name_or_path=google/mt5-large  --dataset_name=ntg  --data_folder=[#input-training-data-path] --output_dir=./results/mT5_large_ntg --per_device_train_batch_size=4 --per_device_eval_batch_size=4 --cache_dir=[#input-previous-model-path] --num_beams=4"

# 8.19
"python -m pip install virtualenv --user && python -m virtualenv /tmp/env_mt5 && . /tmp/env_mt5/bin/activate && python -m pip install --editable .  && pip install -r ./examples/pytorch/summarization/requirements.txt && python -m pip install torch==1.5.0 && python --version && python -u -m torch.distributed.launch --nproc_per_node 4 --use_env examples/pytorch/summarization/run_xglue_no_trainer.py --model_name_or_path=google/mt5-large  --dataset_name=ntg  --data_folder=[#input-training-data-path] --output_dir=./results/mT5_large_ntg --per_device_train_batch_size=1 --per_device_eval_batch_size=4 --cache_dir=[#input-previous-model-path] --num_beams=4 --source_prefix='summarize: '"

# 8.20
"python -m pip install virtualenv --user && python -m virtualenv /tmp/env_mt5 && . /tmp/env_mt5/bin/activate && python -m pip install --editable .  && pip install -r ./examples/pytorch/summarization/requirements.txt && python -m pip install torch==1.5.0 && python --version && python -u -m torch.distributed.launch --nproc_per_node 4 --use_env examples/pytorch/summarization/run_xglue_no_trainer.py --model_name_or_path=google/mt5-small  --dataset_name=ntg  --data_folder=[#input-training-data-path] --output_dir=./results/mT5_small_ntg --per_device_train_batch_size=1 --per_device_eval_batch_size=4 --cache_dir=[#input-previous-model-path] --num_beams=4 --source_prefix='summarize: '"
"python -m pip install virtualenv --user && python -m virtualenv /tmp/env_mt5 && . /tmp/env_mt5/bin/activate && python -m pip install --editable .  && pip install -r ./examples/pytorch/summarization/requirements.txt && python -m pip install torch==1.5.0 && python --version && python -u -m torch.distributed.launch --nproc_per_node 4 --use_env examples/pytorch/summarization/run_xglue_no_trainer.py --model_name_or_path=google/mt5-base  --dataset_name=ntg  --data_folder=[#input-training-data-path] --output_dir=./results/mT5_base_ntg --per_device_train_batch_size=2 --per_device_eval_batch_size=4 --cache_dir=[#input-previous-model-path] --source_prefix='summarize: '"

# 8.23
"python -m pip install virtualenv --user && python -m virtualenv /tmp/env_mt5 && . /tmp/env_mt5/bin/activate && python -m pip install --editable .  && pip install -r ./examples/pytorch/summarization/requirements.txt && python -m pip install torch==1.5.0 && python --version && python -u -m torch.distributed.launch --nproc_per_node 4 --use_env examples/pytorch/summarization/run_xglue_no_trainer.py --model_name_or_path=google/mt5-base  --dataset_name=ntg  --data_folder=[#input-training-data-path] --output_dir=./results/mT5_base_ntg_2 --per_device_train_batch_size=2 --per_device_eval_batch_size=4 --cache_dir=[#input-previous-model-path]"

# 8.25
nohup python -u -m torch.distributed.launch --nproc_per_node 2 --use_env examples/pytorch/text-classification/run_xnli.py  --model_name_or_path="joeddav/xlm-roberta-large-xnli"  --language=de  --output_dir=./results/xnli_de --per_device_eval_batch_size=8 --cache_dir=../ckpt/xlm-roberta-large-xnli --do_predict &> logs/xnli_de.out &
nohup python -u -m torch.distributed.launch --nproc_per_node 2 --use_env examples/pytorch/text-classification/run_xnli.py  --model_name_or_path="joeddav/xlm-roberta-large-xnli"  --language=en  --output_dir=./results/xnli_en --per_device_eval_batch_size=8 --cache_dir=../ckpt/xlm-roberta-large-xnli --do_predict --do_eval --overwrite_cache=True &> logs/xnli_en.out &

# 8.26
nohup python -u -m torch.distributed.launch --nproc_per_node 2 --use_env examples/pytorch/text-classification/run_xlm_roberta_xnli.py  --model_name_or_path="joeddav/xlm-roberta-large-xnli"  --language=en  --output_dir=./results/xnli_en --per_device_eval_batch_size=8 --cache_dir=../ckpt/xlm-roberta-large-xnli --do_predict --do_eval --pad_to_max_length &> logs/xnli_en.out &
CUDA_VISIBLE_DEVICES=0 nohup python -u examples/pytorch/text-classification/run_xlm_roberta_xnli.py  --model_name_or_path="joeddav/xlm-roberta-large-xnli"  --language=en  --output_dir=./results/xnli_en --per_device_eval_batch_size=8 --cache_dir=../ckpt/xlm-roberta-large-xnli --do_predict --do_eval --pad_to_max_length &> logs/xnli_en.out &
CUDA_VISIBLE_DEVICES=0 nohup python -u examples/pytorch/text-classification/run_xlm_roberta_xnli.py  --model_name_or_path="xlm-roberta-large-finetuned-conll03-english"  --language=en  --output_dir=./results/xnli_en --per_device_eval_batch_size=8 --cache_dir=../ckpt/xlm-roberta-large-en --do_predict --do_eval --pad_to_max_length &> logs/xnli_en.out &
CUDA_VISIBLE_DEVICES=1 nohup python -u examples/pytorch/text-classification/run_xnli_each.py  --model_name_or_path="joeddav/xlm-roberta-large-xnli"  --language=en  --output_dir=./results/xnli_en --predict_file=../datasets/DescAndSentence/try_1 --cache_dir=../ckpt/xlm-roberta-large-xnli &> logs/xnli_try.out &
nohup python -u -m torch.distributed.launch --nproc_per_node 2 --use_env examples/pytorch/text-classification/run_xlm_roberta_xnli.py  --model_name_or_path="joeddav/xlm-roberta-large-xnli"  --language=en  --output_dir=./results/xnli_en --per_device_train_batch_size=8 --per_device_eval_batch_size=8 --cache_dir=../ckpt/xlm-roberta-large-xnli --do_train --do_predict --do_eval --pad_to_max_length --num_train_epochs=3 &> logs/xnli_en.out &
nohup python -u examples/pytorch/text-classification/run_xlm_roberta_xnli.py  --model_name_or_path="joeddav/xlm-roberta-large-xnli"  --language=en  --output_dir=./results/xnli_en --per_device_train_batch_size=8 --per_device_eval_batch_size=8 --cache_dir=../ckpt/xlm-roberta-large-xnli --do_train --do_predict --do_eval --pad_to_max_length --num_train_epochs=3 &> logs/xnli_en.out &

"python -m pip install virtualenv --user && python -m virtualenv /tmp/env_xnli && . /tmp/env_xnli/bin/activate && python -m pip install --editable .  && pip install -r ./examples/pytorch/summarization/requirements.txt && python -m pip install torch==1.5.0 && python --version && python -u examples/pytorch/text-classification/run_xnli_each.py  --model_name_or_path=joeddav/xlm-roberta-large-xnli  --language=en  --output_dir=[#output-model-path] --predict_file=[#input-training-data-path] --cache_dir=[#input-previous-model-path] &"
"python -m pip install virtualenv --user && python -m virtualenv /tmp/env_xnli_en && . /tmp/env_xnli_en/bin/activate && python -m pip install --editable .  && pip install -r ./examples/pytorch/text-classification/requirements.txt && python -m pip install torch==1.5.0 && python --version && python -u -m torch.distributed.launch --nproc_per_node 4 --use_env examples/pytorch/text-classification/run_xlm_roberta_xnli.py  --model_name_or_path=joeddav/xlm-roberta-large-xnli  --language=en  --output_dir=./results/xnli_train_en --per_device_train_batch_size=4 --per_device_eval_batch_size=8 --cache_dir=[#input-previous-model-path] --do_train --do_predict --do_eval --pad_to_max_length --num_train_epochs=30 &"
