import os
import qubic
import healpy as hp
import numpy as np
import pylab as plt
import matplotlib as mpl
import sys




mpl.style.use('classic')
name='/Users/hamilton/Python/GitQubicJCH/qubic/qubic/scripts/global_default.dict'

# INSTRUMENT
d = qubic.qubicdict.qubicDict()
d.read_from_file(name)
d['nside']=1024
d['nf_sub'] = 50

s = qubic.QubicScene(d)
q = qubic.QubicMultibandInstrument(d)
Nbfreq_in, nus_edge_in, nus_in, deltas_in, Delta_in, Nbbands_in = qubic.compute_freq(d['filter_nu']/1e9, d['filter_relative_bandwidth'], d['nf_sub']) # Multiband instrument model


q[0].detector


figure(0)
i=0
for i in xrange(len(q)):
	sb = q[i].get_synthbeam(s, 231, detpos=np.array([0.,0.,-0.3]).T)
	clf()
	hp.gnomview(sb/np.max(sb), rot=[0,90], reso=10, min=0,max=0.5, fig=0, 
		title=r'$\nu$ = {0:6.2f} GHz'.format(nus_in[i]))
	savefig('Anim/sb_{}.png'.format(100+i))

