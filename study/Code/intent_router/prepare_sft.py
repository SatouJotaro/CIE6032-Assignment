"""Freeze nested, approximately proportional train subsets; audit token boundaries."""
import collections
import hashlib
import json
import math
from transformers import AutoTokenizer
from prepare import ROOT, DATA, read_rows, save

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def encode_response(tokenizer, row):
    messages=[{'role':'system','content':row['system']},{'role':'user','content':row['instruction']}]
    prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    full=tokenizer.apply_chat_template(messages+[{'role':'assistant','content':row['output']}],tokenize=False,add_generation_prompt=False)
    prefix=tokenizer.encode(prompt,add_special_tokens=False)
    ids=tokenizer.encode(full,add_special_tokens=False)
    assert ids[:len(prefix)]==prefix, 'Prompt boundary mismatch'
    # End response at EOS; exclude trailing chat-template newline from targets.
    eos=ids.index(tokenizer.eos_token_id,len(prefix))
    ids=ids[:eos+1]
    targets=[-100]*len(prefix)+ids[len(prefix):]
    assert len(ids)==len(targets) and 0<len(prefix)<len(ids)
    assert tokenizer.decode(ids[len(prefix):-1])==row['output']
    return ids,targets

def main():
    out=DATA/'sft_v1'
    out.mkdir(exist_ok=True)
    train=read_rows(DATA/'train.jsonl')
    dev=read_rows(DATA/'dev.jsonl')
    labels=json.loads((DATA/'labels.json').read_text())
    pilot=ROOT/'runs/qwen-20261005-040049'
    prompt=(pilot/'prompt.txt').read_text(encoding='utf-8')
    assert hashlib.sha256(prompt.encode()).hexdigest()==json.loads((pilot/'config.json').read_text())['prompt_sha256']
    groups={label:sorted([r for r in train if r['intent']==label],key=lambda r:hashlib.sha256(('sft42:'+str(r['id'])).encode()).hexdigest()) for label in labels}
    # Cover every class once, then choose the least-filled class relative to source size.
    selected=[groups[label][0] for label in labels]
    used={label:1 for label in labels}
    while len(selected)<5000:
        label=min((x for x in labels if used[x]<len(groups[x])),key=lambda x:((used[x]+1)/len(groups[x]),x))
        selected.append(groups[label][used[label]])
        used[label]+=1
    subsets={'train_1000':selected[:1000],'train_5000':selected,'dev':dev}
    lock=json.loads((ROOT/'model_lock.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(ROOT/lock['local_path'],local_files_only=True)
    manifest={'seed_scheme':'SHA256(sft42:id), cover each class then proportional deficit allocation','model':lock,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'source_train_sha256':sha(DATA/'train.jsonl'),'source_dev_sha256':sha(DATA/'dev.jsonl'),'script_sha256':sha(ROOT/'prepare_sft.py'),'cutoff_len':768,'datasets':{}}
    registry={}
    for name,rows in subsets.items():
        data=[{'instruction':r['utt'],'input':'','output':json.dumps({'intent':r['intent']},ensure_ascii=False,separators=(',',':')),'system':prompt} for r in rows]
        lengths=[]
        target_lengths=[]
        for row in data:
            ids,targets=encode_response(tokenizer,row)
            assert len(ids)<=768, 'Would truncate training sample'
            lengths.append(len(ids))
            target_lengths.append(sum(x!=-100 for x in targets))
        path=out/(name+'.json')
        # Never silently replace a previously frozen dataset with different content.
        if path.exists(): assert json.loads(path.read_text(encoding='utf-8'))==data, 'Frozen data differs; use a new version'
        save(path,data)
        manifest['datasets'][name]={'n':len(rows),'ids':[str(r['id']) for r in rows],'label_counts':dict(sorted(collections.Counter(r['intent'] for r in rows).items())),'file_sha256':sha(path),'max_tokens':max(lengths),'total_tokens':sum(lengths),'total_supervised_tokens':sum(target_lengths),'min_supervised_tokens':min(target_lengths)}
        registry['intent_'+name]={'file_name':name+'.json','formatting':'alpaca','columns':{'prompt':'instruction','query':'input','response':'output','system':'system'}}
    assert set(manifest['datasets']['train_1000']['ids']) < set(manifest['datasets']['train_5000']['ids'])
    assert not set(manifest['datasets']['train_5000']['ids']) & set(manifest['datasets']['dev']['ids'])
    save(out/'dataset_info.json',registry)
    save(out/'manifest.json',manifest)
    configs=ROOT/'configs'
    configs.mkdir(exist_ok=True)
    for n in [1000,5000]:
        # Paths are relative to intent_router, including when copied to a GPU host.
        config={'model_name_or_path':lock['local_path'],'trust_remote_code':False,'stage':'sft','do_train':True,'finetuning_type':'lora','lora_rank':8,'lora_alpha':16,'lora_dropout':0.0,'lora_target':'q_proj,v_proj','dataset_dir':'data/sft_v1','dataset':f'intent_train_{n}','eval_dataset':'intent_dev','template':'qwen','cutoff_len':768,'train_on_prompt':False,'packing':False,'preprocessing_num_workers':1,'dataloader_num_workers':0,'output_dir':f'runs/lf-train-{n}','overwrite_output_dir':False,'report_to':'none','push_to_hub':False,'per_device_train_batch_size':1,'gradient_accumulation_steps':8,'per_device_eval_batch_size':1,'learning_rate':0.0001,'num_train_epochs':1.0,'lr_scheduler_type':'cosine','warmup_ratio':0.1,'bf16':True,'gradient_checkpointing':True,'logging_steps':5,'eval_strategy':'epoch','save_strategy':'epoch','seed':42,'data_seed':42}
        save(configs/f'lf_train_{n}.json',config)
    print(json.dumps({k:{a:b for a,b in v.items() if a not in ['ids','label_counts']} for k,v in manifest['datasets'].items()},indent=2))

if __name__=='__main__': main()
