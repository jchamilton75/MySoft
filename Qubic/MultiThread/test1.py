from multi_process2 import parallel_for, parallel_for_chunk
import numpy as np
import time

nprocs = 8
def test(argument):
    a, b, c = common_arguments
    time.sleep(.1)
    return argument**2

def test_chunk(arguments):
    a, b, c = common_arguments
    d = arguments
    output = []
    for args in d:
        time.sleep(0.1)
        output.append(args**2)
    return output

individual_arguments = np.arange(8 * 15)
common_arguments = ['aa', 'bb', 'cc']

nprocs=8
t0 = time.time()
print parallel_for_chunk(test_chunk, individual_arguments, nprocs=nprocs)
print time.time() - t0

t0 = time.time()
print parallel_for(test, individual_arguments, nprocs=nprocs)
print time.time() - t0
