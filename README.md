This repository contains the code to generate n-best hypotheses with a NeMo ASR model and train a LLaMA 2 model for generative error correction. 

The setup closely follows the one implemented in the paper [Whispering-LLaMA](https://aclanthology.org/2023.emnlp-main.618/). You can find the implementation for the same [here](https://github.com/Srijith-rkr/Whispering-LLaMA?tab=readme-ov-file).

## Setup

I used the NeMo container version 23.08 to run this code. You can download the container [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags)

I have also attached the command I used to run my container below:
```
 docker run --gpus all -it --rm -v ./development_code:/workspace/development_code  -v /data/nvidia/:/workspace/development_code/data  --shm-size=8g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 --device=/dev/snd nvcr.io/nvidia/nemo:23.08 
```

This command mounts this repository in `./development_code`

The code changes are implemented on top of commit ID **`7faeee838fb472af45d7547bce56cdd80f69ebbd`** (around December 30th).

 

## Generating N-Best dataset dataset

The paper uses Whisper Tiny (70M) and Large-v2 (1.5B) to generate the n-best dataset. I've added the data files in [data/Whisper_generated](https://github.com/Srijith-rkr/NeMo_WL_merging/tree/wl_merging/data/Whisper_generated) with the necessary data formatting to train the LLaMA 2 model in NeMo.

I've also included the script to generate your own n-best dataset using Whisper, as done in the paper, in this [repo](https://github.com/Srijith-rkr/Whisper-LLaMA-Nemo)

You can also to refer [data/NeMo_generated/generating_data_with_Nemo_backend.py](https://github.com/Srijith-rkr/NeMo_WL_merging/blob/wl_merging/data/NeMo_generated/generating_data_with_Nemo_backend.py) on how to use NeMo's native ASR model (`stt_en_fastconformer_transducer_large`) to generate the same.

## Training LLaMA-2 model 

[This](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/llama2peft.html) tutorial is very helpful on setting up the LLaMA-2 model for training. 
**I will be adding more details tomorrow**

## Inference with LLaMA-2 model 

## Results
## N-best code explained 
## Other helpful tips 
    
