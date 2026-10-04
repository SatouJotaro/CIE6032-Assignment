---
library_name: peft
license: other
base_model: models/Qwen2.5-0.5B-Instruct
tags:
- base_model:adapter:models/Qwen2.5-0.5B-Instruct
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: factory-check-20261005-041933
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# factory-check-20261005-041933

This model is a fine-tuned version of [models/Qwen2.5-0.5B-Instruct](https://huggingface.co/models/Qwen2.5-0.5B-Instruct) on the intent_train_1000 dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: constant
- training_steps: 2

### Training results



### Framework versions

- PEFT 0.17.1
- Transformers 4.57.1
- Pytorch 2.14.1+cpu
- Datasets 4.0.0
- Tokenizers 0.22.2