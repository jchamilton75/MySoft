from __future__ import division
import multiprocessing
import numpy as np


### Generic consumer class
class Consumer(multiprocessing.Process):
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                #print '%s: Exiting' % proc_name
                self.task_queue.task_done()
                break
            #print '%s: %s' % (proc_name, next_task)
            answer = next_task()
            self.task_queue.task_done()
            self.result_queue.put(answer)
        return

### generic class for task to be performed by consumers
class Task(object):
    def __init__(self, functname,arguments,index):
        self.functname=functname
        self.args = arguments
        self.index = index
    def __call__(self):
        return(self.functname(self.args,self.index))


### This is the function to be called by user
def parallel_loop(functname, args, istart, istop, nthreads,emptyoutput):
    #number of calls to be performed
    ntodo=istop-istart+1
    #number of multithread shots
    nshots=ntodo//nthreads+1
    ntot=nshots*nthreads
    indices=np.arange(ntot)
    indices[indices>ntodo-1]=ntodo-1
    lines=np.reshape(indices,(nshots,nthreads))

    # now loop on the shots (each shot consists in running the function nthread times in //)
    for num in np.arange(nshots):
        # initiate multithreading
        tasks=multiprocessing.JoinableQueue()
        results = multiprocessing.Queue()
        # start consumers
        num_consumers = nthreads
        consumers = [ Consumer(tasks, results) for i in xrange(num_consumers) ]
        for w in consumers:
            w.start()
        # Enqueue jobs
        num_jobs = nthreads
        for i in np.arange(num_jobs):
            tasks.put(Task(functname,args,lines[num,i]))
        # poison them when finished
        for i in np.arange(num_jobs):
            tasks.put(None)
        print('Just ran the the following indices:')
        print(lines[num,np.arange(num_jobs)])
        tasks.join()
        while num_jobs:
            result = results.get()
            bla=result[0]
            num=result[1]
            emptyoutput[:,num,num:]=bla[:,num:]
            emptyoutput[:,num:,num]=bla[:,num:]
            num_jobs -= 1

    return(emptyoutput)

    


    