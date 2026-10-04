"""Explicit scaled dot-product attention and causal-mask checks on CPU. Requires torch."""
import math
import torch


def attention(q,k,v,causal=False):
    scores=q@k.transpose(-2,-1)/math.sqrt(q.shape[-1])
    if causal:
        if q.shape[-2]!=k.shape[-2]:
            raise ValueError('This exercise uses equal-length causal self-attention.')
        future=torch.triu(torch.ones(q.shape[-2],k.shape[-2],dtype=torch.bool),diagonal=1)
        scores=scores.masked_fill(future,float('-inf'))
    weights=torch.softmax(scores,dim=-1)
    return weights@v,weights


def run():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    q=torch.randn(2,5,8,requires_grad=True)
    k=torch.randn(2,5,8,requires_grad=True)
    v=torch.randn(2,5,6,requires_grad=True)
    out,weights=attention(q,k,v,True)
    assert out.shape==(2,5,6)
    assert torch.allclose(weights.sum(-1),torch.ones(2,5),atol=1e-6)
    assert torch.count_nonzero(torch.triu(weights,diagonal=1))==0
    altered_v=v.detach().clone();altered_v[:,4]+=100
    altered_out,_=attention(q,k,altered_v,True)
    assert torch.allclose(out[:,:4],altered_out[:,:4],atol=1e-5), 'Future value leaked into earlier outputs'
    out.square().mean().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in [q,k,v])
    print('Q,K,V:',tuple(q.shape),tuple(k.shape),tuple(v.shape))
    print('Output:',tuple(out.shape))
    print('First batch causal weights:\n',weights[0].detach())
    print('PASS: shape, row sums, future mask, no future-value leakage, finite Q/K/V gradients.')
    print('Try: remove the mask and rerun the future-value experiment; explain what changes.')


if __name__=='__main__':
    run()
