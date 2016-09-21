import multiprocessing
import numpy as np


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
            #print '******* %s: %s' % (proc_name, next_task)
            answer = next_task()
            self.task_queue.task_done()
            self.result_queue.put(answer)
        return


class Task(object):
    def __init__(self, i, thefunct, arguments, common_arguments):
        self.i = i
        self.thefunct = thefunct
        self.arguments = arguments
        self.common_arguments = common_arguments
        #print('in Task - done arguments {}'.format(i))
    def __call__(self):
        aa=self.thefunct(self.i,self.arguments, self.common_arguments)
        return([aa,self.i])


def parallel_for(thefunct, individual_arguments, common_arguments, nprocs=8):
    nbtot = len(individual_arguments)
    ninchunk = nbtot // nprocs ## integer division
    arguments_chunk = []
    indices = []
    for i in xrange(nprocs-1):
        indices.append(np.arange(i*ninchunk,(i+1)*ninchunk))
        arguments_chunk.append(individual_arguments[i*ninchunk:(i+1)*ninchunk])
    arguments_chunk.append(individual_arguments[(nprocs-1)*ninchunk:])
    indices.append(np.arange((nprocs-1)*ninchunk, nbtot))
    
    # initiate multithreading
    tasks=multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()
    # start consumers
    num_consumers = nprocs
    consumers = [ Consumer(tasks, results)
                  for i in xrange(num_consumers) ]
    for w in consumers:
        w.start()
    for i in np.arange(num_consumers):
        thearguments = arguments_chunk[i]
        print('I am consumer #{} - Running for {} iterations'.format(i,len(thearguments)))
        tasks.put(Task(i, thefunct, thearguments, common_arguments))
        #print('Task {} is put'.format(i))
    # poison them when finished
    for i in np.arange(num_consumers):
        #print('now killing {}'.format(i))
        tasks.put(None)
    #print('now joining queues')
    tasks.join()
    #print('queues joined')
    
    # Get results
    output = []
    output_num = []
    while num_consumers:
        result = results.get()
        output.append(result[0])
        output_num.append(result[1])    
        num_consumers -= 1

    ### Reordering results
    toreturn=[]
    for n in xrange(len(output_num)):
        bla = output[output_num.index(n)]
        for elts in bla:
            toreturn.append(elts)
    return(toreturn)
    
    
    


