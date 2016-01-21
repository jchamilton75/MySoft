import test_cpu as test
import numpy as np
import socket
machine=socket.gethostname()
nb=30
num=np.logspace(3,4,nb).astype(int)

all=np.zeros((4,nb))

for i in np.arange(nb):
    print(i)
    all[:,i]=test.doit(num[i])
    print(all[:,i])

A=np.array([np.log10(num),np.ones(len(num))])
w0 = linalg.lstsq(A.T,np.log10(all[0,:]))[0]
w1 = linalg.lstsq(A.T,np.log10(all[1,:]))[0]
w2 = linalg.lstsq(A.T,np.log10(all[2,:]))[0]
w3 = linalg.lstsq(A.T,np.log10(all[3,:]))[0]



clf()
plot(num,all[0,:],'ko',label='Matrix Allocation: slope={0:.2f} t10000={1:.2f} s'.format(w0[0],10**w0[1]*10000**w0[0]))
plot(num,10**(w0[0]*np.log10(num)+w0[1]),'k-')
plot(num,all[1,:],'bo',label='Vector Allocation: slope={0:.2f} t10000={1:.5f} s'.format(w1[0],10**w1[1]*10000**w1[0]))
plot(num,10**(w1[0]*np.log10(num)+w1[1]),'b-')
plot(num,all[2,:],'go',label='Matrix.Vector Calculation: slope={0:.2f} t10000={1:.2f} s'.format(w2[0],10**w2[1]*10000**w2[0]))
plot(num,10**(w2[0]*np.log10(num)+w2[1]),'g-')
plot(num,all[3,:],'ro',label='Matrix.Matrix Calculation: slope={0:.2f} t10000={1:.2f} s'.format(w3[0],10**w3[1]*10000**w3[0]))
plot(num,10**(w3[0]*np.log10(num)+w3[1]),'r-')
xscale('log')
yscale('log')
legend(loc='upper left')
xlabel('Number of elements')
ylabel('Time in seconds')
xlim(np.min(num),np.max(num))
ylim(1e-5,1e3)
title(machine)
savefig('toto_laptop.png')
 
