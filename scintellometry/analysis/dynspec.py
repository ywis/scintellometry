import numpy as np

n = f.sum(0)[..., (0,3)].sum(-1) / ic.sum(0)
n_median = np.median(n, axis=1)
nn = n / n_median[:, np.newaxis] - 1.

profile = n[200:350].sum(0)
profile = profile / np.median(profile) -1.
profile[profile < 0.] = 0.
profile /= profile.sum()

nt = f[..., (0,3)].sum(-1) / ic

dyn = ((nt / np.median(nt, axis=2)[..., np.newaxis] - 1.) * profile).sum(-1)
vmin = dyn.mean() - 1*dyn.std()
vmax = dyn.mean() + 5*dyn.std()

plt.imshow(dyn.T, aspect='auto', interpolation='nearest', origin='lower',
           cmap=plt.get_cmap('binary'), vmin=vmin, vmax=vmax,
           extent=(0., 37., 200., 400.))
plt.xlabel('t (min)')
plt.ylabel('f (MHz)')
plt.title('PSR B1957')
