from __future__ import print_function
from multi_process import parallel_for, parallel_for_chunk
from qubic import create_random_pointings, QubicAcquisition, QubicScene
import numpy as np
import time

sampling = create_random_pointings([0, 90], 1000, 10)
scene = QubicScene()
acq = QubicAcquisition(150, sampling, scene)


def test(argument):
    #print(argument)
    time.sleep(np.random.rand()/100)
    return acq[argument].instrument.detector.center[0]

individual_arguments = np.arange(992)
#common_arguments = [scene]

#### test scalar version
def scalar_for(funct, args):
    out = []
    for a in args:
        out.append(funct(a))
    return out

res = scalar_for(test, individual_arguments)    
resp = parallel_for(test, individual_arguments,nprocs=8)

result = np.array(parallel_for(test, individual_arguments, nprocs=8))




%timeit res = scalar_for(test, individual_arguments)    
%timeit resp = parallel_for(test, individual_arguments,nprocs=8)


