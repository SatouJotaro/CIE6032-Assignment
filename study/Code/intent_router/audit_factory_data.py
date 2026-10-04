"""Check all exported records with the actual Factory data pipeline (no training)."""
from verify_factory import ROOT,get_train_args,load_tokenizer,get_template_and_fix_tokenizer,get_dataset
from prepare_sft import encode_response
from prepare import save
import json

def main():
    reports={}
    for size in [1000,5000]:
        config=json.loads((ROOT/f'configs/lf_train_{size}.json').read_text())
        config.update(use_cpu=True,bf16=False,fp16=False,output_dir=str(ROOT/'runs/factory-data-audit'),cache_dir=str(ROOT/'.hf-lf'))
        ma,da,ta,_,_=get_train_args(config)
        module=load_tokenizer(ma)
        tokenizer=module['tokenizer']
        template=get_template_and_fix_tokenizer(tokenizer,da)
        datasets=get_dataset(template,ma,da,ta,stage='sft',**module)
        for key,filename in [('train_dataset',f'train_{size}'),('eval_dataset','dev')]:
            source=json.loads((ROOT/f'data/sft_v1/{filename}.json').read_text(encoding='utf-8'))
            actual=datasets[key]
            assert len(source)==len(actual)
            max_tokens=0
            for row,encoded in zip(source,actual):
                ids,labels=encode_response(tokenizer,row)
                assert encoded['input_ids'][:len(ids)]==ids
                assert encoded['labels'][:len(labels)]==labels
                suffix=encoded['input_ids'][len(ids):]
                assert not suffix or tokenizer.decode(suffix)=='\n'
                assert encoded['labels'][len(labels):]==suffix
                max_tokens=max(max_tokens,len(encoded['input_ids']))
            reports[f'{size}/{key}']={'rows':len(actual),'all_prefixes_and_loss_masks_match':True,'max_tokens':max_tokens}
    save(ROOT/'runs/factory-data-audit.json',reports)
    print(json.dumps(reports,indent=2))

if __name__=='__main__': main()
