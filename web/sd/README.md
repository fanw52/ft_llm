

## 使用步骤

1. 下载模型依赖

可以通过shell脚本批量下载：

同一个指令执行多次，是为了防止网络断开而停止下载
```shell
python tools/download_model.py --repo_id stabilityai/stable-diffusion-xl-base-1.0 --local_dir /data1/pretrained_models/stable-diffusion-xl-base-1.0
python tools/download_model.py --repo_id stabilityai/stable-diffusion-xl-base-1.0 --local_dir /data1/pretrained_models/stable-diffusion-xl-base-1.0
python tools/download_model.py --repo_id stabilityai/stable-diffusion-xl-base-1.0 --local_dir /data1/pretrained_models/stable-diffusion-xl-base-1.0
python tools/download_model.py --repo_id stabilityai/stable-diffusion-xl-base-1.0 --local_dir /data1/pretrained_models/stable-diffusion-xl-base-1.0
python tools/download_model.py --repo_id stabilityai/stable-diffusion-xl-base-1.0 --local_dir /data1/pretrained_models/stable-diffusion-xl-base-1.0
python tools/download_model.py --repo_id stabilityai/stable-diffusion-xl-base-1.0 --local_dir /data1/pretrained_models/stable-diffusion-xl-base-1.0
python tools/download_model.py --repo_id stabilityai/stable-diffusion-xl-base-1.0 --local_dir /data1/pretrained_models/stable-diffusion-xl-base-1.0


python tools/download_model.py --repo_id stabilityai/stable-diffusion-xl-refiner-1.0 --local_dir /data1/pretrained_models/stable-diffusion-xl-refiner-1.0
python tools/download_model.py --repo_id stabilityai/stable-diffusion-xl-refiner-1.0 --local_dir /data1/pretrained_models/stable-diffusion-xl-refiner-1.0
python tools/download_model.py --repo_id stabilityai/stable-diffusion-xl-refiner-1.0 --local_dir /data1/pretrained_models/stable-diffusion-xl-refiner-1.0
python tools/download_model.py --repo_id stabilityai/stable-diffusion-xl-refiner-1.0 --local_dir /data1/pretrained_models/stable-diffusion-xl-refiner-1.0
python tools/download_model.py --repo_id stabilityai/stable-diffusion-xl-refiner-1.0 --local_dir /data1/pretrained_models/stable-diffusion-xl-refiner-1.0
python tools/download_model.py --repo_id stabilityai/stable-diffusion-xl-refiner-1.0 --local_dir /data1/pretrained_models/stable-diffusion-xl-refiner-1.0
python tools/download_model.py --repo_id stabilityai/stable-diffusion-xl-refiner-1.0 --local_dir /data1/pretrained_models/stable-diffusion-xl-refiner-1.0

```

2. shell脚本执行后，会在/data/pretrained_models下生成两个文件夹，stable-diffusion-xl-base-1.0，stable-diffusion-xl-refiner-1.0，如果保存到其他位置，需要修改脚本demo_sdxl_base_refiner.py中的base_model_path，以及refiner_model_path的路径

3. 启动： streamlit run demo_sdxl_base_refiner.py