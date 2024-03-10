This repository contains the code to generate n-best hypotheses with a NeMo ASR model (stt_en_fastconformer_transducer_large) and train a LLaMA 2 model for generative error correction. 

The setup closely follows the one implemented in the paper [Whispering-LLaMA](https://aclanthology.org/2023.emnlp-main.618/). You can find the implementation for the same [here](https://github.com/Srijith-rkr/Whispering-LLaMA?tab=readme-ov-file).

## Setup

I used the NeMo container version 23.08 to run this code. You can download the container [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags)

I have also attached the command I used to run my container below:
```
 docker run --gpus all -it --rm -v ./development_code:/workspace/development_code  -v /data/nvidia/:/workspace/development_code/data  --shm-size=8g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 --device=/dev/snd nvcr.io/nvidia/nemo:23.08 
```

This command mounts this repository in `./development_code`

The code changes are implemented on top of the NeMo commit ID **`7faeee838fb472af45d7547bce56cdd80f69ebbd`** (around December 30th).

 

## Generating N-Best dataset dataset

The paper uses Whisper Tiny (70M) and Large-v2 (1.5B) to generate the n-best dataset. I've added the data files in [data/Whisper_generated](https://github.com/Srijith-rkr/NeMo_WL_merging/tree/wl_merging/data/Whisper_generated) with the necessary data formatting to train the LLaMA 2 model in NeMo.

I've also included the script to generate your own n-best dataset using Whisper, as done in the paper, in this [repo](https://github.com/Srijith-rkr/Whisper-LLaMA-Nemo)

You can also to refer [data/NeMo_generated/generating_data_with_Nemo_backend.py](https://github.com/Srijith-rkr/NeMo_WL_merging/blob/wl_merging/data/NeMo_generated/generating_data_with_Nemo_backend.py) on how to use NeMo's native ASR model (`stt_en_fastconformer_transducer_large`) to generate the same.

## Training and Inference on LLaMA-2 model 

[This](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/llama2peft.html) tutorial is very helpful on setting up the LLaMA-2 model for training. Just modifying the dataset path on the config file will be enough. 

To run inference, I recommend using the checkpoint with the lowest loss rather than the final checkpoint. I have modified the LLaMA eval code a little bit to run inferenece with checkpoints. The path to the file is  `NeMo_WL_mergin/examples/nlp/language_modeling/tuning/megatron_gpt_peft_eval_w_ckpt.py` and it uses the config file `examples/nlp/language_modeling/tuning/conf/megatron_gpt_peft_eval_w_ckpt.yaml`. I have commented '--update' on all the important parts of the config you will have to change so that you can find it with Cntrl+F. 

## Results
The below table represents the one-best (greedy decoding) Word Error Rate with Whisper Tiny (70M) and Large (1.5B) models. 
|Running inference with Whisper (1-Best)| | |
|:----|:----|:----|
|Dataset|Whisper Large|Whisper Tiny|
|Entertainment |15.86|24.25|
|Science and Technology|15.85|19.51|

The below table represents the 15-best (Temperature sampling) Word Error Rate. The Whisper Tiny and Large N-best hypothises were generated as mentioned on the [paper](https://aclanthology.org/2023.emnlp-main.618/) (Section 4.2) "For each audio clip, we generate 200 hypotheses using a top-k value of 200 and a randomly selected temperature between the range of [0.7, 0.8]. Subsequently, we filter out redundant sentences and
select the top 15 with the highest log probability". You can find a easy implementation of the same in this repo [Whisper-LLaMA-Nemo](https://github.com/Srijith-rkr/Whisper-LLaMA-Nemo).

The stt_en_fastconformer_transducer_large N-best hypothesis are generated using `data/NeMo_generated/generating_data_with_Nemo_backend.py`. I have explained how on the below section. 
|Oracle | | | |
|:----|:----|:----|:----|
|Dataset|Whisper Large|Whisper Tiny|stt_en_fastconformer_transducer_large|
|Entertainment |21.78|28.22|35.7|
|Science and Technology|18.02|23.93|32.64|


The below table represents the Word Error Rate of the fine-tuned LLaMA model. Column 1 and 2 are results from the paper and were trained using a LLaMA 1 variant (Alpaca). Column 3 was trained on NeMo's LLaMA 2 training pipeline. 
|Fine-tuning with LLaMA| | | |
|:----|:----|:----|:----|
|Dataset|Whisper Large|Whisper Tiny|stt_en_fastconformer_transducer_large|
|Entertainment |15.59|21.61|**20.75**|
|Science and Technology|**12.77**|18.02|-|


## N-best code explained 
I have modified `nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py` to replace greedy-decoding with temperature sampling to adapt NeMo's native stt_en_fastconformer_transducer_large model to generate N-best hypothesis. Please refer to commit **1d2e364** to see the code changes. I recommed trying out different temperature values to further optimize oracle N-best performance. 

    
