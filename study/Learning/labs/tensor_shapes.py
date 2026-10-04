"""CPU CNN / residual / ViT patch-shape checks. Requires torch, no downloads."""
import torch
from torch import nn


def run():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    x=torch.randn(2,3,32,32)
    conv=nn.Conv2d(3,16,3,padding=1)
    y=conv(x)
    params=sum(p.numel() for p in conv.parameters())
    print('Conv:',tuple(x.shape),'->',tuple(y.shape),'parameters=',params)
    assert y.shape==(2,16,32,32) and params==448
    pooled=nn.MaxPool2d(2)(y)
    print('Pool:',tuple(pooled.shape))
    assert pooled.shape==(2,16,16,16)
    # Residual downsampling block: align both spatial size and channel count.
    main=nn.Conv2d(16,32,3,stride=2,padding=1)(pooled)
    skip=nn.Conv2d(16,32,1,stride=2)(pooled)
    residual=main+skip
    print('Residual branches:',tuple(main.shape),tuple(skip.shape),'->',tuple(residual.shape))
    assert residual.shape==(2,32,8,8)
    image=torch.randn(2,3,224,224)
    embedding=nn.Conv2d(3,64,16,stride=16)(image)
    tokens=embedding.flatten(2).transpose(1,2)
    print('Patch tokens:',tuple(tokens.shape),'(without class token)')
    assert tokens.shape==(2,196,64)
    print('Try: change stride/padding, predict the shape BEFORE running; change patch 16 -> 8.')


if __name__=='__main__':
    run()
