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
nohup python -u -m torch.distributed.launch --nproc_per_node 2 examples/pytorch/summarization/run_xglue_no_trainer.py  --model_name_or_path 'microsoft/xprophetnet-large-wiki100-cased-xglue-ntg'  --dataset_name=ntg --data_folder=../datasets/xglue_full_dataset/sampled_NTG  --output_dir=./results/xpro_ntg --per_device_train_batch_size=1 --per_device_eval_batch_size=2 --cache_dir=../ckpt/xprophtnet_ntg &> logs/xpro_ntg_2.out &
CUDA_VISIBLE_DEVICES=1,0 accelerate launch examples/pytorch/summarization/run_xglue_no_trainer.py  --model_name_or_path 'microsoft/xprophetnet-large-wiki100-cased-xglue-ntg'  --dataset_name=ntg --data_folder=../datasets/xglue_full_dataset/sampled_NTG  --output_dir=./results/xpro_ntg --per_device_train_batch_size=1 --per_device_eval_batch_size=2 --cache_dir=../ckpt/xprophtnet_ntg &> logs/xpro_ntg_2.out &
