"""Small, real CPU classification experiment. Requires torch; no data downloads.
Run: python mlp_cpu.py --epochs 240 --lr 0.02 --seed 7
Outputs metrics.json and learning_curve.csv under labs/results by default.
Change one setting at a time. Validation is NOT a held-out final test set.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import torch
from torch import nn


def run(args):
    if args.epochs < 1 or args.lr <= 0:
        raise ValueError('epochs and lr must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    # Synthetic balanced rings; all tensors stay on CPU.
    n = 480
    y = torch.arange(n) % 2
    angle = 2*math.pi*torch.rand(n)
    radius = .65 + .65*y + .10*torch.randn(n)
    x = torch.stack([radius*torch.cos(angle), radius*torch.sin(angle)], dim=1)
    order = torch.randperm(n)
    train_idx, val_idx = order[:360], order[360:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    # Fit preprocessing using only the training split.
    mean, std = x_train.mean(0), x_train.std(0).clamp_min(1e-6)
    x_train, x_val = (x_train-mean)/std, (x_val-mean)/std
    model = nn.Sequential(nn.Linear(2,32), nn.ReLU(), nn.Linear(32,32), nn.ReLU(), nn.Linear(32,2))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    rows = []
    for epoch in range(args.epochs+1):
        if epoch > 0:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_train)
            loss = criterion(logits,y_train)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_logits, val_logits = model(x_train), model(x_val)
            train_loss = criterion(train_logits,y_train).item()
            val_loss = criterion(val_logits,y_val).item()
            train_acc = (train_logits.argmax(1)==y_train).float().mean().item()
            val_acc = (val_logits.argmax(1)==y_val).float().mean().item()
        row = dict(epoch=epoch,train_loss=train_loss,val_loss=val_loss,train_acc=train_acc,val_acc=val_acc)
        rows.append(row)
        if epoch%40==0 or epoch==args.epochs:
            print(f'epoch {epoch:3d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val acc {val_acc:.3f}')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'learning_curve.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    report={'seed':args.seed,'epochs':args.epochs,'learning_rate':args.lr,'torch':torch.__version__,
            'device':'cpu','train_samples':360,'val_samples':120,'parameters':sum(p.numel() for p in model.parameters()),
            'initial':rows[0],'final':rows[-1],
            'limitation':'Synthetic task, one random split, validation only; not a benchmark or generalization guarantee.'}
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    assert torch.isfinite(torch.tensor([r['train_loss'] for r in rows])).all(), 'Non-finite loss'
    print('Saved:',args.output.resolve())
    print('Explain: why raw logits? what changes if zero_grad is omitted? how was leakage avoided?')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=240)
    p.add_argument('--lr',type=float,default=.02)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'results')
    run(p.parse_args())
