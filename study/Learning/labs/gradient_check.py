"""Numerical vs analytic backprop for a two-layer tanh classifier. Standard library only.
Run: python gradient_check.py
Exercise: derive grads() yourself before reading its implementation.
"""
import math


def forward(theta, x, target):
    # W1: two output rows, two input columns; b1: 2; W2: 2x2; b2: 2.
    h = [math.tanh(sum(theta[2*j+i] * x[i] for i in range(2)) + theta[4+j]) for j in range(2)]
    z = [sum(theta[6+2*k+j] * h[j] for j in range(2)) + theta[10+k] for k in range(2)]
    m = max(z)
    es = [math.exp(v-m) for v in z]
    p = [v/sum(es) for v in es]
    return -math.log(p[target]), h, p


def grads(theta, x, target):
    loss, h, p = forward(theta, x, target)
    dz = [p[k] - (k == target) for k in range(2)]
    dh = [sum(theta[6+2*k+j] * dz[k] for k in range(2)) for j in range(2)]
    da = [dh[j] * (1-h[j]**2) for j in range(2)]
    return ([da[j]*x[i] for j in range(2) for i in range(2)] + da
            + [dz[k]*h[j] for k in range(2) for j in range(2)] + dz)


def run():
    theta = [.1, -.2, .3, .4, .05, -.1, .2, -.1, -.3, .5, .1, -.05]
    x, target, eps = [.6, -1.2], 1, 1e-5
    analytic = grads(theta, x, target)
    numerical = []
    for i in range(len(theta)):
        plus, minus = theta.copy(), theta.copy()
        plus[i] += eps
        minus[i] -= eps
        numerical.append((forward(plus,x,target)[0]-forward(minus,x,target)[0])/(2*eps))
    errors = [abs(a-n) for a,n in zip(analytic,numerical)]
    print('index  analytical      numerical       absolute error')
    for i,(a,n,e) in enumerate(zip(analytic,numerical,errors)):
        print(f'{i:>5}  {a: .9f}   {n: .9f}   {e:.3e}')
    print(f'Max absolute error: {max(errors):.3e}')
    assert max(errors) < 1e-7, 'Gradient mismatch: inspect the chain rule and indexing.'
    loss = forward(theta,x,target)[0]
    updated = [v-.1*g for v,g in zip(theta,analytic)]
    after = forward(updated,x,target)[0]
    print(f'One update: loss {loss:.6f} -> {after:.6f}')
    assert after < loss
    print('Try: change eps; change x/target; deliberately omit tanh derivative, then compare.')


if __name__ == '__main__':
    run()
