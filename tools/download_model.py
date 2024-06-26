import argparse

from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument('--repo_id', default="baichuan", type=str)
parser.add_argument('--local_dir', default="/data/pretrained_models/tmp", type=str)
parser.add_argument('--test', action="store_true")

args = parser.parse_args()

# FlagAlpha/Llama2-Chinese-13b-Chat
# python tools/download_model.py --repo_id  OpenBuddy/openbuddy-llama-65b-v8-bf16 --local_dir /data/pretrained_models/openbuddy-llama-65b-v8-bf16

'''
python tools/download_model.py --repo_id  stabilityai/stable-diffusion-2-1 --local_dir /data/pretrained_models/stable-diffusion-2-1
python tools/download_model.py --repo_id stabilityai/stablelm-tuned-alpha-7b --local_dir /data/pretrained_models/stablelm-tuned-alpha-7b
python tools/download_model.py --repo_id runwayml/stable-diffusion-v1-5 --local_dir /data/pretrained_models/stable-diffusion-v1-5
'''
# snapshot_download(repo_id=args.repo_id, local_dir=args.local_dir)
snapshot_download(repo_id=args.repo_id, local_dir=args.local_dir)
