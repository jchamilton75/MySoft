import pylab
import numpy

#### Copy and rename TD calfiles
#### if TD
cp /Users/hamilton/Python/MySoft/Qubic/TD/TD_CalQubic_HornArray_v4.fits /Users/hamilton/Python/GitQubicJCH/qubic/qubic/calfiles/CalQubic_HornArray_v5.fits
cp /Users/hamilton/Python/MySoft/Qubic/TD/TD_CalQubic_DetArray_v3.fits /Users/hamilton/Python/GitQubicJCH/qubic/qubic/calfiles/CalQubic_DetArray_v4.fits
#### if not TD
rm -f /Users/hamilton/Python/GitQubicJCH/qubic/qubic/calfiles/CalQubic_HornArray_v5.fits
rm -f /Users/hamilton/Python/GitQubicJCH/qubic/qubic/calfiles/CalQubic_DetArray_v4.fits



from qubic import QubicInstrument
inst = QubicInstrument(filter_nu=150e9)

xx,yy,zz = inst.detector.center.T
index_det = inst.detector.index

pylab.figure(figsize=(12,10))
for i in xrange(len(inst.detector)):
    pylab.text(xx[i]-0.0012,yy[i],'{}'.format(index_det[i]), fontsize=12, color='r')
pylab.title('detector index',fontsize=32)
pylab.xlabel('x coordinate (m)',fontsize=32)
pylab.ylabel('y coordinate (m)',fontsize=32)
inst.detector.plot()


centers = inst.horn.center[:,0:2] 
col = inst.horn.column
row = inst.horn.row
pylab.figure(figsize=(12,10))
for i in xrange(len(centers)):
    pylab.text(centers[i,0]-0.006, centers[i,1], 'c{0:}'.format(col[i]), color='r',fontsize=12)
    pylab.text(centers[i,0]+0.001, centers[i,1], 'r{0:}'.format(row[i]), color='b',fontsize=12)
pylab.xlabel('x coordinate (m)',fontsize=32)
pylab.ylabel('y coordinate (m)',fontsize=32)
inst.horn.plot()





