"""Strict common evaluator. Prediction JSONL: {id, output: JSON string}."""
import argparse
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report
from prepare import read_rows, save

def evaluate(gold, predictions, labels):
    ids = [str(r['id']) for r in gold]
    pred_ids = [str(r['id']) for r in predictions]
    if len(set(ids)) != len(ids) or len(set(pred_ids)) != len(pred_ids):
        raise ValueError('Duplicate IDs')
    if set(ids) != set(pred_ids):
        raise ValueError('Prediction IDs must exactly match gold IDs')
    by_id = {str(r['id']):r['output'] for r in predictions}
    predicted, valid_json, valid_schema = [], 0, 0
    for id_ in ids:
        try:
            value = json.loads(by_id[id_])
            valid_json += 1
        except (ValueError, TypeError):
            value = None
        valid = isinstance(value, dict) and set(value) == {'intent'} and isinstance(value['intent'], str) and value['intent'] in labels
        valid_schema += int(valid)
        predicted.append(value['intent'] if valid else '__INVALID__')
    truth = [r['intent'] for r in gold]
    return {
        'n':len(gold), 'accuracy':accuracy_score(truth,predicted),
        'macro_f1':f1_score(truth,predicted,labels=labels,average='macro',zero_division=0),
        'json_valid_rate':valid_json/len(gold), 'schema_valid_rate':valid_schema/len(gold),
        'invalid_count':len(gold)-valid_schema,
        'per_class':classification_report(truth,predicted,labels=labels,output_dict=True,zero_division=0),
        'errors':[{'id':r['id'],'text':r['utt'],'gold':r['intent'],'predicted':p} for r,p in zip(gold,predicted) if r['intent'] != p]
    }

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--gold',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--labels',type=Path,default=Path(__file__).parent/'data/labels.json')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    save(a.out,evaluate(read_rows(a.gold),read_rows(a.predictions),json.loads(a.labels.read_text(encoding='utf-8'))))
