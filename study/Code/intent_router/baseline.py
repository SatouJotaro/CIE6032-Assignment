"""Fixed CPU baseline; train-only vocabulary/IDF, dev-only evaluation."""
import datetime
import hashlib
import json
import platform
import time
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from prepare import ROOT, DATA, read_rows, save, normalize
from evaluate import evaluate

def main():
    train, dev = read_rows(DATA/'train.jsonl'), read_rows(DATA/'dev.jsonl')
    labels=json.loads((DATA/'labels.json').read_text(encoding='utf-8'))
    config={'analyzer':'char','ngram_range':[1,3],'min_df':2,'sublinear_tf':True,'C':1.0,'random_state':42,'max_iter':5000}
    model=make_pipeline(TfidfVectorizer(analyzer='char',ngram_range=(1,3),min_df=2,sublinear_tf=True),LinearSVC(C=1.0,random_state=42,max_iter=5000,dual='auto'))
    start=time.perf_counter()
    model.fit([r['utt'] for r in train],[r['intent'] for r in train])
    fit_seconds=time.perf_counter()-start
    start=time.perf_counter()
    pred=model.predict([r['utt'] for r in dev])
    predict_seconds=time.perf_counter()-start
    predictions=[{'id':r['id'],'output':json.dumps({'intent':str(v)},ensure_ascii=False)} for r,v in zip(dev,pred)]
    metrics=evaluate(dev,predictions,labels)
    train_texts={normalize(r['utt']) for r in train}
    clean=[r for r in dev if normalize(r['utt']) not in train_texts]
    clean_ids={str(r['id']) for r in clean}
    clean_metrics=evaluate(clean,[r for r in predictions if str(r['id']) in clean_ids],labels)
    out=ROOT/'runs'/datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True,exist_ok=False)
    save(out/'metrics.json',metrics)
    save(out/'dev_unseen_text_metrics.json',clean_metrics)
    (out/'dev_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in predictions),encoding='utf-8')
    report={'model':'char TF-IDF + LinearSVC','split':'dev','test_evaluated':False,'config':config,'python':platform.python_version(),'sklearn':sklearn.__version__,'fit_seconds':fit_seconds,'dev_predict_seconds':predict_seconds,'features':len(model[0].vocabulary_),'accuracy':metrics['accuracy'],'macro_f1':metrics['macro_f1'],'dev_unseen_text_n':len(clean),'dev_unseen_text_accuracy':clean_metrics['accuracy'],'dev_unseen_text_macro_f1':clean_metrics['macro_f1'],'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['baseline.py','evaluate.py','prepare.py']},'data_sha256':{p:hashlib.sha256((DATA/p).read_bytes()).hexdigest() for p in ['train.jsonl','dev.jsonl','labels.json']},'executed_by':'assistant; user has not yet independently reproduced this run'}
    save(out/'report.json',report)
    save(ROOT/'runs/latest.json',{'run':out.name})
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()

