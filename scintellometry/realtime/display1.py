"""To start time series
ssh algonquin@192.168.1.33
cd ../scratch/time-series/
# rm *pow*.dat
./run.timeseries

Need to cross-mount /home from pen-node5
mount -tnfs 192.168.1.5:/home /mnt/aro-home
cd /mnt/aro-home/scratch/time-series
ipython
run -i ~/packages/scintellometry/scintellometry/realtime/display.py
"""
from __future__ import division, print_function
import numpy as np
import matplotlib.pylab as plt
from time import sleep
import sys

plt.ion()
maxlen = 100  # factor of 10

SAMPLE_FILE = 'quarter_sample_power.1.dat'
START_READ_AT_END = True
PLOT_AT_END = False

nset = {'half': 2, 'full': 1, 'quar': 4, 'onee': 8}[SAMPLE_FILE[:4]]

samples = []
times = []
with open(SAMPLE_FILE,'r') as fs:
    if START_READ_AT_END:
        fs.seek(0,2)
    while True:
        # wait for a new time sample to appear
        line = fs.readline()
        cnt = 0
        while line == '' or line[:5] != '2013 ':
            if line == '':
                sys.stdout.write("\rwaiting for next timestamp; " +
                                 "count={}".format(cnt))
                sys.stdout.flush()
                cnt += 1
                sleep(0.2)
            line = fs.readline()
        sys.stdout.write("\rGot timestamp {}".format(line))
        sys.stdout.flush()
        # convert to float seconds since start of day
        tl = line.split()
        if len(tl) < 7:
            continue
        times += [(float(tl[3])*60+float(tl[4]))*60+float(tl[5])+float(tl[6])]
        # read corresponding samples
        newset = []
        for i in range(nset):
            sample = fs.readline()
            cnt = 0
            while sample == '':
                sys.stdout.write("\rwaiting for next sample; count={}".format(
                    cnt))
                sys.stdout.flush()
                cnt += 1
                if cnt > 100:
                    break
                sleep(0.1)
                sample = fs.readline()
            try:
                newset += [float(sample)]
            except:
                cnt = 1000
                break
        if cnt > 100:
            continue

        samples += newset

        if len(times) == 1:
            continue
        if len(times) > maxlen:
            times.pop(0)
            for i in range(nset):
                samples.pop(0)

        plt.clf()
        t = np.array(times)
        td = t[1]-t[0]
        t = (t[:,np.newaxis]+td*np.arange(0,nset)/nset).flatten()
        s = np.array(samples)
        plt.plot(t, s)
        if len(samples) >= 20:
            l = len(samples) % 20
            tav = t[l:].reshape(-1,10).mean(1)
            sav = s[l:].reshape(-1,10).mean(1)
            plt.plot(tav, sav)
        plt.draw()
