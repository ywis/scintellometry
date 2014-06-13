import numpy as np

f2_03 = f2[...,(0,3)].sum(-1)
f2_cln = f2_03[:80]
ic_cln = ic[:80]

f = f2_cln.sum(0)/ic_cln.sum(0)
ok = f.std(1) < 0.003

prof = (f2_cln * ok[:, np.newaxis]).sum((0,1))/ic_cln.sum((0,1))
prof -= prof.min()

tall = f2_cln/ic_cln
dyn = ((tall-tall.min(-1, keepdims=True))*prof).sum(-1)/prof.sum()
