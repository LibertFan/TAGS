# NSGDC

Some codes in this repo are copied/modified from opensource implementations made available by
[UNITER](https://github.com/ChenRocks/UNITER),
[PyTorch](https://github.com/pytorch/pytorch),
[HuggingFace](https://github.com/huggingface/transformers),
[OpenNMT](https://github.com/OpenNMT/OpenNMT-py),
and [Nvidia](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch).
The image features are extracted using [BUTD](https://github.com/peteanderson80/bottom-up-attention).


## Requirements
This is following UNITER. We provide Docker image for easier reproduction. Please install the following:
  - [nvidia driver](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation) (418+), 
  - [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) (19.03+), 
  - [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#quickstart).

Our scripts require the user to have the [docker group membership](https://docs.docker.com/install/linux/linux-postinstall/)
so that docker commands can be run without sudo.
We only support Linux with NVIDIA GPUs. We test on Ubuntu 18.04 and V100 cards.
We use mixed-precision training hence GPUs with Tensor Cores are recommended.

## Image-Text Retrieval
### Download Data
```
bash scripts/download_itm.sh $PATH_TO_STORAGE
```
The new txt_db file in  https://drive.google.com/drive/folders/1ZOK3jlcgGRifz8iw2-5vL89rIJcoYU3D?usp=sharing. Please download the txt_db file to replace the original one.

### Launch the Docker Container
```bash
# docker image should be automatically pulled
source launch_container.sh $PATH_TO_STORAGE/txt_db $PATH_TO_STORAGE/img_db \
$PATH_TO_STORAGE/finetune $PATH_TO_STORAGE/pretrained
```

In case you would like to reproduce the whole preprocessing pipeline.

The launch script respects $CUDA_VISIBLE_DEVICES environment variable.
Note that the source code is mounted into the container under `/src` instead 
of built into the image so that user modification will be reflected without
re-building the image. (Data folders are mounted into the container separately
for flexibility on folder structures.)


### Image-Text Retrieval (Flickr30k)
```
# Train wit the base setting
bash run_cmds/tran_pnsgd_base_flickr.sh
bash run_cmds/tran_pnsgd2_base_flickr.sh

# Train wit the large setting
bash run_cmds/tran_pnsgd_large_flickr.sh
bash run_cmds/tran_pnsgd2_large_flickr.sh
```

### Image-Text Retrieval (COCO)
```
# Train wit the base setting
bash run_cmds/tran_pnsgd_base_coco.sh
bash run_cmds/tran_pnsgd2_base_coco.sh

# Train wit the large setting
bash run_cmds/tran_pnsgd_large_coco.sh
bash run_cmds/tran_pnsgd2_large_coco.sh
```

### Run Inference
```
bash run_cmds/inf_nsgd.sh
```

## Results

Our models achieve the following performance.

### MS-COCO
<table>
	<tr>
	    <th rowspan="2">Model</th>
	    <th colspan="3">Image-to-Text</th>
	    <th colspan="3">Text-to-Image</th>  
	</tr >
	<tr>
	    <td>R@1</td>
	    <td>R@5</td>
	    <td>R@110</td>
	    <td>R@1</td>
	    <td>R@5</td>
	    <td>R@10</td>
	</tr>
	<tr>
	    <td>NSGDC-Base</td>
	    <td>66.6</td>
        <td>88.6</td>
        <td>94.0</td>
        <td>51.6</td>
        <td>79.1</td>
        <td>87.5</td>
	</tr>
	<tr>
	    <td>NSGDC-Large</td>
	    <td>67.8</td>
        <td>89.6</td>
        <td>94.2</td>
        <td>53.3</td>
        <td>80.0</td>
        <td>88.0</td>
	</tr>
</table>

### Flickr30K


<table>
	<tr>
	    <th rowspan="2">Model</th>
	    <th colspan="3">Image-to-Text</th>
	    <th colspan="3">Text-to-Image</th>  
	</tr >
	<tr>
	    <td>R@1</td>
	    <td>R@5</td>
	    <td>R@110</td>
	    <td>R@1</td>
	    <td>R@5</td>
	    <td>R@10</td>
	</tr>
	<tr>
	    <td>NSGDC-Base</td>
	    <td>87.9</td>
        <td>98.1</td>
        <td>99.3</td>
        <td>74.5</td>
        <td>93.3</td>
        <td>96.3</td>
	</tr>
	<tr>
	    <td>NSGDC-Large</td>
	    <td>90.6</td>
        <td>98.8</td>
        <td>99.1</td>
        <td>77.3</td>
        <td>94.3</td>
        <td>97.3</td>
	</tr>
</table>
