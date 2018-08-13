import numpy as np
from matplotlib import *
from pylab import *

jobid, nnodes, nptg, hh, mm, ss, mem = np.loadtxt('/obs/jhamilton/Qubic/SpectroImager/stats.txt').T

seconds = ss+60*mm+3600*hh

uni_nnodes = np.unique(nnodes)
uni_nptg = np.unique(nptg)

clf()
for i in xrange(len(uni_nnodes)):
	nn = uni_nnodes[i]
	ok = nnodes == nn
	order = np.argsort(nptg[ok])
	print(ok)
	subplot(2,1,1)
	plot(nptg[ok][order], seconds[ok][order]*1000/nptg[ok][order], 'o-', label = 'Nnodes={}'.format(nn))
	xlabel('Nptg')
	ylabel('Time(sec) norm. to 1000')
	xscale('log')
	xlim(900,20000)
	yscale('log')
	subplot(2,1,2)
	plot(nptg[ok][order], mem[ok][order]/1e6*1000/nptg[ok][order], 'o-', label = nn)
	xlabel('Nptg')
	ylabel('Mem(GB) norm. to 1000')
	xscale('log')
	xlim(900,20000)
	yscale('log')
legend(frameon=False, fontsize=8)



clf()
for i in xrange(len(uni_nptg)):
	nn = uni_nptg[i]
	ok = nptg == nn
	order = np.argsort(nnodes[ok])
	print(ok)
	subplot(2,1,1)
	plot(nnodes[ok][order], seconds[ok][order]*1000/nn, 'o-', label = 'Nptg={}'.format(nn))
	xlabel('Nodes')
	ylabel('Time(sec) norm. to 1000')
	xscale('log')
	xlim(0.9,6)
	yscale('log')
	subplot(2,1,2)
	plot(nnodes[ok][order], mem[ok][order]/1e6*1000/nn, 'o-', label = nn)
	xlabel('NNodes')
	ylabel('Mem(GB) norm. to 1000')
	xscale('log')
	xlim(0.9,6)
	yscale('log')
legend(frameon=False, fontsize=8)



