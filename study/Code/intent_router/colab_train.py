"""Run in Colab only: GPU probe or one explicit training size, with Drive outputs."""
import argparse
import json
import os
import time
from pathlib import Path
import torch
from transformers import TrainerCallback
from llamafactory.train.tuner import run_exp

class Measure(TrainerCallback):
    def __init__(self,out): self.out=out
    def on_train_begin(self,args,state,control,**kwargs):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); self.start=time.perf_counter()
    def on_train_end(self,args,state,control,**kwargs):
        torch.cuda.synchronize()
        report={'gpu':torch.cuda.get_device_name(),'seconds':time.perf_counter()-self.start,'steps':state.global_step,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'torch':torch.__version__,'cuda':torch.version.cuda}
        (self.out/'gpu_measurement.json').write_text(json.dumps(report,indent=2))
        print(report,flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--size',type=int,choices=[1000,5000],default=1000)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',type=Path)
    a=p.parse_args()
    assert torch.cuda.is_available(), 'Choose a GPU runtime first; will not silently train on CPU.'
    out=a.output.resolve()
    if a.resume:
        assert a.resume.is_dir() and (a.resume/'trainer_state.json').exists()
    else:
        assert not out.exists(), 'Use a new output folder, or explicitly resume a checkpoint.'
    out.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path(f'configs/lf_train_{a.size}.json').read_text())
    bf16=torch.cuda.is_bf16_supported()
    config.update(bf16=bf16,fp16=not bf16,output_dir=str(out),save_strategy='steps',save_steps=25,save_total_limit=2,save_only_model=False,disable_gradient_checkpointing=False)
    if a.smoke:
        config.update(max_steps=10,max_samples=80,eval_strategy='no',warmup_ratio=0.0)
        config.pop('eval_dataset',None)
    if a.resume:
        previous=json.loads((out/'run_config.json').read_text())
        comparable={k:v for k,v in previous.items() if k!='resume_from_checkpoint'}
        assert comparable==config, 'Resume config differs; do not mix experiments.'
        config['resume_from_checkpoint']=str(a.resume)
    (out/'run_config.json').write_text(json.dumps(config,indent=2))
    Path('colab_last_output.txt').write_text(str(out))
    run_exp(config,callbacks=[Measure(out)])

if __name__=='__main__': main()
