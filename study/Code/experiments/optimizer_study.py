"""CPU optimizer study. No downloads. Paired initializations and batch order across configs.
python Code/experiments/optimizer_study.py --epochs 100 --seeds 7 17 27
Produces a fresh timestamped run directory. Not a real-world benchmark.
"""
import argparse, copy, csv, hashlib, json, math, platform, statistics, time
from pathlib import Path
import torch
from torch import nn

CONFIGS=[('sgd_lr003','sgd',.03,0.),('sgd_lr030','sgd',.3,0.),('momentum_lr003','sgd',.03,.9),('adam_lr003','adam',.03,0.),('adam_lr0003','adam',.003,0.)]

def dataset():
    g=torch.Generator().manual_seed(20261005)
    n=1200; y=torch.arange(n)%2; angle=2*math.pi*torch.rand(n,generator=g)
    radius=.65+.55*y+.19*torch.randn(n,generator=g)
    x=torch.stack([radius*torch.cos(angle),radius*torch.sin(angle)],1)
    idx=torch.randperm(n,generator=g); tr,va=idx[:900],idx[900:]
    mean=x[tr].mean(0);std=x[tr].std(0).clamp_min(1e-6)
    return (x[tr]-mean)/std,y[tr],(x[va]-mean)/std,y[va]

def make_model():return nn.Sequential(nn.Linear(2,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,2))
def make_optimizer(model,kind,lr,momentum):
    return torch.optim.Adam(model.parameters(),lr=lr) if kind=='adam' else torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum)
def update(model,opt,x,y):
    model.train();opt.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(model(x),y);loss.backward()
    grad=math.sqrt(sum(p.grad.detach().square().sum().item() for p in model.parameters() if p.grad is not None))
    opt.step();return grad
def evaluate(model,x,y):
    model.eval()
    with torch.no_grad():
        z=model(x);return nn.functional.cross_entropy(z,y).item(),(z.argmax(1)==y).float().mean().item()

def resume_check(initial,x,y,output):
    model=make_model();model.load_state_dict(initial);opt=make_optimizer(model,'adam',.003,0)
    for i in range(5):update(model,opt,x[i*64:(i+1)*64],y[i*64:(i+1)*64])
    checkpoint=output/'resume_checkpoint.pt'
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':5},checkpoint)
    update(model,opt,x[320:384],y[320:384])
    restored=make_model();other=make_optimizer(restored,'adam',.003,0)
    saved=torch.load(checkpoint,weights_only=True);restored.load_state_dict(saved['model']);other.load_state_dict(saved['optimizer'])
    update(restored,other,x[320:384],y[320:384])
    error=max((a-b).abs().max().item() for a,b in zip(model.parameters(),restored.parameters()))
    assert error<1e-7
    return {'next_update_max_abs_error':error,'scope':'Deterministic CPU fixed next batch, no dropout; full random/dataloader-state recovery not tested.'}

def main(args):
    if args.epochs<1 or args.batch<1 or args.threads<1:raise ValueError('epochs/batch/threads must be positive')
    torch.set_num_threads(args.threads)
    root=Path(__file__).resolve().parents[1]/'runs';root.mkdir(exist_ok=True)
    output=root/time.strftime('optimizer-%Y%m%d-%H%M%S');output.mkdir(exist_ok=False)
    x,y,xv,yv=dataset(); criterion=nn.CrossEntropyLoss()
    # Warm up tensor kernels; timings still depend on CPU load and scheduling.
    warm=make_model();wo=torch.optim.SGD(warm.parameters(),lr=.01)
    for _ in range(3):update(warm,wo,x[:64],y[:64])
    runs=[];curve=[];first_initial=None
    for seed in args.seeds:
        torch.manual_seed(seed);initial=copy.deepcopy(make_model().state_dict())
        if first_initial is None:first_initial=initial
        order_gen=torch.Generator().manual_seed(seed+1000)
        orders=[torch.randperm(len(x),generator=order_gen) for _ in range(args.epochs)]
        for name,kind,lr,momentum in CONFIGS:
            model=make_model();model.load_state_dict(initial);opt=make_optimizer(model,kind,lr,momentum)
            start=time.perf_counter();training_seconds=0.;steps=0;rows=[];hit=None
            for epoch in range(args.epochs+1):
                grads=[]
                if epoch:
                    begin=time.perf_counter()
                    for batch in orders[epoch-1].split(args.batch):
                        grads.append(update(model,opt,x[batch],y[batch]));steps+=1
                    training_seconds+=time.perf_counter()-begin
                train_loss,train_acc=evaluate(model,x,y);val_loss,val_acc=evaluate(model,xv,yv)
                row={'config':name,'seed':seed,'epoch':epoch,'step':steps,'train_seconds':training_seconds,'wall_seconds':time.perf_counter()-start,'train_loss':train_loss,'val_loss':val_loss,'train_acc':train_acc,'val_acc':val_acc,'grad_norm_mean':statistics.mean(grads) if grads else 0.}
                if not math.isfinite(train_loss):raise RuntimeError(f'Non-finite loss: {name}, seed {seed}')
                rows.append(row);curve.append(row)
                if hit is None and val_acc>=args.threshold:hit={'epoch':epoch,'step':steps,'wall_seconds':row['wall_seconds']}
            with (output/f'{name}-seed{seed}.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
            runs.append({'config':name,'seed':seed,'optimizer':kind,'lr':lr,'momentum':momentum,'initial':rows[0],'final':rows[-1],'first_threshold_hit':hit,'best_val_acc':max(r['val_acc'] for r in rows)})
            print(name,seed,'val_acc',round(rows[-1]['val_acc'],4),'wall_s',round(rows[-1]['wall_seconds'],2),flush=True)
    summary=[]
    for name,*_ in CONFIGS:
        group=[r for r in runs if r['config']==name]
        acc=[r['final']['val_acc'] for r in group];wall=[r['final']['wall_seconds'] for r in group]
        summary.append({'config':name,'val_acc_mean':statistics.mean(acc),'val_acc_sample_std':statistics.stdev(acc) if len(acc)>1 else None,'wall_seconds_mean':statistics.mean(wall),'threshold_hits':sum(r['first_threshold_hit'] is not None for r in group),'runs':len(group)})
    report={'generated_at':time.strftime('%Y-%m-%d %H:%M:%S'),'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'device':'cpu','threads':args.threads},'settings':vars(args),'dataset':{'name':'synthetic noisy rings','train':900,'validation':300,'data_seed':20261005,'final_test_set':False,'preprocessing':'training mean/std only'},'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'paired_controls':'Same fixed dataset/split; for each seed, identical initial parameters and per-epoch batch order across all configurations.','timing_scope':'Training time includes gradient norm collection; wall time includes evaluation. CPU measurements are order/load dependent, not hardware benchmarks.','resume_check':resume_check(first_initial,x,y,output),'summary':summary,'runs':runs,'curves':curve,'limitations':['Synthetic low-dimensional task, not an LLM or real dataset benchmark.','A few fixed settings, not equal-budget exhaustive optimizer tuning.','Seeds vary initialization and batch order, not data split; std describes these seeds only.','Validation threshold is a predefined observation, not a guarantee of stable future performance.','Agent executed this baseline; learner must reproduce, modify and explain it before claiming personal experience.']}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (root/'latest.json').write_text(json.dumps({'run':output.name,'report':str((output/'report.json').relative_to(root.parent))},indent=2),encoding='utf-8')
    print('REPORT',output/'report.json',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch',type=int,default=64);p.add_argument('--seeds',type=int,nargs='+',default=[7,17,27]);p.add_argument('--threads',type=int,default=1);p.add_argument('--threshold',type=float,default=.9);main(p.parse_args())
