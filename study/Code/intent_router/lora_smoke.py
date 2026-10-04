"""Two CPU optimizer steps on one training example; integration check, not evaluation."""
import datetime
import hashlib
import json
import time
import torch
import peft
import transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import LoraConfig,get_peft_model,PeftModel
from prepare import ROOT, DATA, save
from prepare_sft import encode_response

def frozen_hash(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if not p.requires_grad:
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def main():
    torch.set_num_threads(6)
    torch.manual_seed(42)
    out=ROOT/'runs'/('lora-smoke-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    out.mkdir()
    lock=json.loads((ROOT/'model_lock.json').read_text())
    path=ROOT/lock['local_path']
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    row=json.loads((DATA/'sft_v1/train_1000.json').read_text(encoding='utf-8'))[0]
    ids,targets=encode_response(tokenizer,row)
    x=torch.tensor([ids]); y=torch.tensor([targets])
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.float32,attn_implementation='sdpa')
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(task_type='CAUSAL_LM',r=8,lora_alpha=16,lora_dropout=0.0,target_modules=['q_proj','v_proj']))
    trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert trainable and all('lora_' in n for n in trainable)
    initial={n:p.detach().clone() for n,p in trainable.items()}
    frozen_before=frozen_hash(model)
    optimizer=torch.optim.AdamW(trainable.values(),lr=1e-4,weight_decay=0)
    steps=[]
    for step in range(2):
        start=time.perf_counter()
        model.train(); optimizer.zero_grad()
        loss=model(input_ids=x,attention_mask=torch.ones_like(x),labels=y).loss
        assert torch.isfinite(loss)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(list(trainable.values()),1.0)
        assert torch.isfinite(norm) and norm>0
        optimizer.step()
        steps.append({'step':step+1,'loss':loss.item(),'grad_norm_before_clip':norm.item(),'seconds':time.perf_counter()-start})
        print(steps[-1],flush=True)
    assert frozen_hash(model)==frozen_before, 'Frozen base weights changed'
    delta=max((p.detach()-initial[n]).abs().max().item() for n,p in trainable.items())
    assert delta>0
    model.eval()
    with torch.inference_mode(): before=model(input_ids=x,labels=y).loss.item()
    model.save_pretrained(out/'adapter',safe_serialization=True)
    base=model.unload()
    restored=PeftModel.from_pretrained(base,out/'adapter',local_files_only=True).eval()
    with torch.inference_mode(): after=restored(input_ids=x,labels=y).loss.item()
    assert abs(before-after)<1e-6
    save(out/'report.json',{'scope':'two steps, one TRAIN sample; no generalization claim','model':lock,'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'steps':steps,'tokens':len(ids),'supervised_tokens':sum(v!=-100 for v in targets),'trainable_parameters':sum(p.numel() for p in trainable.values()),'frozen_weights_unchanged':True,'max_adapter_update':delta,'reload_loss_difference':abs(before-after),'final_train_example_loss':after,'script_sha256':hashlib.sha256((ROOT/'lora_smoke.py').read_bytes()).hexdigest(),'sft_manifest_sha256':hashlib.sha256((DATA/'sft_v1/manifest.json').read_bytes()).hexdigest(),'test_used':False,'framework':'standalone PEFT; NOT a LLaMA-Factory validation'})
    save(ROOT/'runs/latest_lora_smoke.json',{'run':out.name})
    print('Verified and saved:',out,flush=True)

if __name__=='__main__': main()
