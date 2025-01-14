# MamMIL: Multiple Instance Learning for Whole Slide Images with State Space Models
Accepted by IEEE BIBM 2024.

# Prepare the Instance Features

## If you want to begin from WSIs

* We use the WSIs from [BRACS](https://www.bracs.icar.cnr.it/), [Camelyon 16](https://camelyon17.grand-challenge.org/Data/), and [TCGA-LUAD](https://portal.gdc.cancer.gov/). You can download the WSIs from their public links.

* After downloading the WSIs, please use the [CLAM repository](https://github.com/mahmoodlab/CLAM) to pre-process the WSIs and extract the features. The features should be put into the `./data` directory.

* Then, use the `h5toPyG.ipynb` notebook to produce the required Graph representation of the WSIs.

## Or you can use the pre-extracted featrues

* Please download the extracted features from [Baidu Disk](https://pan.baidu.com/s/1gjC1qymw40xpkDZay_SeRA?pwd=dxgt), or from [OneDrive](https://1drv.ms/f/s!AhnWr7i0cJHtmqY_ey6LgcG9EtDGSA).

* Then, unzip it to the root directory of this repository. The final architecture should be like:

    ```
    -- data
    |__ BRACS/BRACS_512_at_level0
    |____h5_files
    |____PyG_files
    |____vim
    |______h5_files
    |______PyG_files
    |____vmamba
    |...
    ```

# Environment

## Tested on
* Ubuntu 18.04 & 20.04

* 1x RTX 3090 GPU

* CUDA 11.8

* Python 3.10.14

* Pytorch 2.1.2

* Pytorch-lightning 1.6.3

* torchmetrics 0.9.3

* torch_geometric 2.5.2

* causal-conv1d 1.4.0

* pip 24.0

## Prepare the environment with PIP/CONDA
Please first create a virtual environment with `python 3.10.14` and install the dependencies specified in `requirements.in`.

Then, run the following commands to install additional dependencies.

```bash
pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

cd ./mamba && pip3 install .

pip3 install causal-conv1d==1.4.0
```

**Important:** If you encounter any issues during the prepation of the environment, please check the package versions.

## Prepare the environment with Docker
Tested on Docker version 20.10.17, build 100c701

```bash
DOCKER_BUILDKIT=1 docker build -t mammil:train .
```

NOTE: you need to set up the proxy by yourself if you are in China Mainland.

# Train & Test

## If you are using PIP/CONDA

Just modify `/usr/local/bin/python3` in `run.sh` to `python` and run

```bash
bash run.sh
```

## If you are using Docker
Run

```bash
docker run --gpus "device=0" --rm -it --shm-size 8G -v /path/to/your/data:/opt/app/data -v /path/to/your/logs:/opt/app/logs mammil:train
```

Make sure you have give `777` access to the `./logs` directory.

Finally, the results will be in `./logs`

# Reproduce the results in the paper
Since the `selective_scan` operation in Mamba is not deterministic, you may get different results from the paper if you train the model from scratch.

If you want to fully reproduce the results in our paper, you can download pre-extracted features and the docker container from [Baidu Disk](https://pan.baidu.com/s/1bmNPM6d6effUSYC-d2iuNQ?pwd=rmt9) or [OneDrive](https://1drv.ms/u/s!AhnWr7i0cJHtmqYGFfHzTAp5A342Yw?e=BiwoqP). Put it in this repository, and then run

```bash
docker load < mammil-v1_20241031_172354.tar.gz

docker run --gpus "device=0" --rm -it --shm-size 8G -v /path/to/your/data:/opt/app/data -v /path/to/your/outputs:/outputs mammil:v1
```

Make sure you have give `777` access to the `./outputs` directory.

# Citation
If you find our work helpful, please cite our paper:
```text
@INPROCEEDINGS{10822552,
  author={Fang, Zijie and Wang, Yifeng and Zhang, Ye and Wang, Zhi and Zhang, Jian and Ji, Xiangyang and Zhang, Yongbing},
  booktitle={2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)}, 
  title={MamMIL: Multiple Instance Learning for Whole Slide Images with State Space Models}, 
  year={2024},
  volume={},
  number={},
  pages={3200-3205},
  keywords={Degradation;Deep learning;Analytical models;Pathology;Codes;Biological system modeling;Transformers;Graph neural networks;Complexity theory;Biomedical imaging;Multiple Instance Learning;State Space Models;Whole Slide Images},
  doi={10.1109/BIBM62325.2024.10822552}}
```

