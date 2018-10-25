import qubic
import healpy as hp


scene = qubic.QubicScene(256)
inst = qubic.instrument.QubicInstrument(filter_nu=150e9)

clf()
inst.detector.plot()
vertex = inst.detector.vertex[..., :2]
detcenters = inst.detector.center[..., :2]
for i in xrange(992): 
	text(detcenters[i,0]-0.001, detcenters[i,1],i, fontsize=7)

detnum = 0
sb = inst[detnum].get_synthbeam(scene)[0]


hp.gnomview(sb, rot=[0,90], reso=5,
            title='Synthesized Beam')



