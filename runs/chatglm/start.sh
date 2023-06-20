#PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 python ./tmp/infer_wx_int4.py
#PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 python ./tmp/infer_wx_fp16.py
#PYTHONPATH=. CUDA_VISIBLE_DEVICES=2,3 python ./tmp/infer_general_int4.py
#PYTHONPATH=. CUDA_VISIBLE_DEVICES=2,3 python ./tmp/infer_general_fp16.py

bash ./runs/train_chatglm_lora_p2p_v1.1.0.sh
bash ./runs/train_chatglm_lora_p2p_v1.1.1.sh
bash ./runs/train_chatglm_lora_p2p_v1.1.2.sh
