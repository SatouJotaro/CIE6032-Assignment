"""Validate actual LLaMA-Factory preprocessing, then run two bounded CPU steps."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ['HF_HOME']=str(ROOT/'.hf-lf')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_DATASETS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import datetime
import importlib.metadata
import json
import hashlib
import torch
from prepare import save
from prepare_sft import encode_response
from llamafactory.hparams import get_train_args
from llamafactory.model import load_tokenizer
from llamafactory.data import get_template_and_fix_tokenizer,get_dataset
from llamafactory.train.tuner import run_exp

def main():
    torch.set_num_threads(6)
    out=ROOT/'runs'/('factory-check-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    out.mkdir()
    original=json.loads((ROOT/'configs/lf_train_1000.json').read_text())
    config=dict(original)
    config.update(use_cpu=True,bf16=False,fp16=False,max_steps=2,max_samples=4,
                  gradient_accumulation_steps=1,gradient_checkpointing=False,
                  eval_strategy='no',save_strategy='no',logging_steps=1,
                  output_dir=str(out),dataloader_pin_memory=False,
                  warmup_ratio=0.0,lr_scheduler_type='constant',
                  cache_dir=str(ROOT/'.hf-lf'),disable_tqdm=True)
    config.pop('eval_dataset')
    save(out/'config.json',config)
    model_args,data_args,training_args,finetuning_args,_=get_train_args(config)
    module=load_tokenizer(model_args)
    tokenizer=module['tokenizer']
    template=get_template_and_fix_tokenizer(tokenizer,data_args)
    datasets=get_dataset(template,model_args,data_args,training_args,stage='sft',**module)
    source=json.loads((ROOT/'data/sft_v1/train_1000.json').read_text(encoding='utf-8'))[:4]
    data=datasets['train_dataset']
    assert len(data)==4
    details=[]
    for row,encoded in zip(source,data):
        ids,targets=encode_response(tokenizer,row)
        # Factory qwen format may include a final newline after EOS; require exact
        # common span and permit ONLY that explicit suffix, never prompt leakage.
        actual_ids=encoded['input_ids']; actual_labels=encoded['labels']
        assert actual_ids[:len(ids)]==ids
        assert actual_labels[:len(targets)]==targets
        suffix=actual_ids[len(ids):]
        assert not suffix or tokenizer.decode(suffix)=='\n'
        assert actual_labels[len(targets):]==suffix
        details.append({'tokens':len(actual_ids),'supervised_tokens':sum(v!=-100 for v in actual_labels),'suffix':tokenizer.decode(suffix)})
    save(out/'preprocessing_check.json',{'matched':True,'rows':details,'scope':'first four train records through actual Factory loader'})
    print('PREPROCESSING MATCHED',flush=True)
    run_exp(config)
    metrics=json.loads((out/'train_results.json').read_text())
    assert (out/'adapter_model.safetensors').is_file()
    versions={name:importlib.metadata.version(name) for name in ['llamafactory','transformers','peft','datasets','accelerate','torch','tokenizers','huggingface-hub']}
    save(out/'verification.json',{'passed':True,'versions':versions,'training_metrics':metrics,'preprocessing_checked':4,'optimizer_steps':2,'base_config_sha256':hashlib.sha256((ROOT/'configs/lf_train_1000.json').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'CPU integration check only, not full training or GPU performance','test_used':False})
    save(ROOT/'runs/latest_factory_check.json',{'run':out.name})
    print('FACTORY VERIFIED',out,flush=True)

if __name__=='__main__': main()
