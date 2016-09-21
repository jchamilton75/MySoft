from pathos.pools import ProcessPool as Pool
from pyoperators.utils import split
import itertools
import time

TIMEOUT = 86400

def parallel_for(func, args, nprocs=8):
    pool = Pool(nprocs)
    return pool.amap(func, args).get(TIMEOUT)

def parallel_for_chunk(func, args, nprocs=8):
    pool = Pool(nprocs)
    slices = tuple(split(len(args), nprocs))
    def wrapper(islice):
        return func(args[slices[islice]])
    out = pool.amap(wrapper, xrange(len(slices))).get(TIMEOUT)
    return list(itertools.chain(*out))
