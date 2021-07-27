from sympy import expand, symbols, integrate
import torch

def wallenius_noncentral_hypergeometric_distribution_pmf(x, w):
    assert len(x) == len(w)
    D = sum((1-xi)*wi for xi, wi in zip(x, w))
    coefs = [wi/D for wi in w]
    t = symbols('t')
    exprs = [(1-t**ci)**xi for ci,xi in zip(coefs, x)]
    expanded = expand('*'.join(['('+str(ei)+')' for ei in exprs]))
    pmf = integrate(expanded, (t, 0, 1))
    return pmf


### Test cases ###
def manual_2(i,j,p):
    return p[i]*p[j]/(1-p[i]) + p[j]*p[i]/(1-p[j])

def manual_3(x, w):
    ps = []
    for n,xi in enumerate(x):
        w_ = w[:]
        if xi != 1:
            continue
        wi = w_.pop(n)
        w_ = [w/(1-wi) for w in w_]
        x_ = x[:n] + x[n+1:]
        i,j = [n for n,x in enumerate(x_) if x==1]
        # print(w_, x_, i, j)
        p = wi * manual_2(i, j, w_)
        ps.append(p)
    return sum(ps)

# Verified
x = [1,0,1,0]               # Draws in sample
w = [0.4, 0.3, 0.2, 0.1]    # Weights
print(wallenius_noncentral_hypergeometric_distribution_pmf(x, w))
manual_2(0, 2, w)

x = [1,0,1,0,0,1]
w = [0.2, 0.1, 0.1, 0.3, 0.1, 0.2]
print(manual_3(x, w))
print(wallenius_noncentral_hypergeometric_distribution_pmf(x, w))

z = range(7)
x = [1 if n in z else 0 for n in range(20)]
w = torch.randn(20).softmax(-1).tolist()
print(wallenius_noncentral_hypergeometric_distribution_pmf(x, w))

