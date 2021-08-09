# 8.9
python examples/pytorch/summarization/run_summarization.py  --model_name_or_path t5-small  --do_train  --do_eval  --dataset_name cnn_dailymail  --dataset_config "3.0.0"  --source_prefix "summarize: "  --output_dir ./results/t5_cnn_dailymail --per_device_train_batch_size=4 --per_device_eval_batch_size=4 --overwrite_output_dir --predict_with_generate
python examples/pytorch/summarization/run_xglue.py  --model_name_or_path 'microsoft/xprophetnet-large-wiki100-cased-xglue-ntg'  --do_eval  --do_test --dataset_name xglue  --dataset_config ntg   --output_dir ./results/xlm_pro_ntg_test --per_device_train_batch_size=4 --per_device_eval_batch_size=4 --overwrite_output_dir --predict_with_generate --cache_dir=../ckpt/xprophtnet_ntg

