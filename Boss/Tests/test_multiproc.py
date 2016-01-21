
# from http://pymotw.com/2/multiprocessing/communication.html

# simple test running 4 matrix inversions in //
# unfortunately I cannot recover the return values
import multiprocessing

def worker(num,sz):
    print 'worker ',num, 'starts'
    mat=np.random.rand(sz,sz)
    b=np.linalg.det(mat)
    print 'result :',b
    return(num**2)

size=1000

if __name__ == '__main__':
    jobs = []
    for i in range(5):
        p = multiprocessing.Process(target=worker,args=(i,size))
        jobs.append(p)
        p.start()




#### another example
import multiprocessing

class MyFancyClass(object):
    
    def __init__(self, name):
        self.name = name
    
    def do_something(self):
        proc_name = multiprocessing.current_process().name
        print 'Doing something fancy in %s for %s!' % (proc_name, self.name)


def worker(q):
    obj = q.get()
    obj.do_something()


if __name__ == '__main__':
    queue = multiprocessing.Queue()

    p = multiprocessing.Process(target=worker, args=(queue,))
    p.start()
    
    queue.put(MyFancyClass('Fancy Dan'))
    
    # Wait for the worker to finish
    queue.close()
    queue.join_thread()
    p.join()
    






### more complex example
import multiprocessing
import time

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
                print '%s: Exiting' % proc_name
                self.task_queue.task_done()
                break
            print '%s: %s' % (proc_name, next_task)
            answer = next_task()
            self.task_queue.task_done()
            self.result_queue.put(answer)
        return


class Task(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def __call__(self):
        time.sleep(10) # pretend to take some time to do the work
        return '%s * %s = %s' % (self.a, self.b, self.a * self.b)
    def __str__(self):
        return '%s * %s' % (self.a, self.b)


if __name__ == '__main__':
    # Establish communication queues
    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()
    
    # Start consumers
    num_consumers = multiprocessing.cpu_count() * 2
    print 'Creating %d consumers' % num_consumers
    consumers = [ Consumer(tasks, results)
                  for i in xrange(num_consumers) ]
    for w in consumers:
        w.start()
    
    # Enqueue jobs
    num_jobs = 16
    for i in xrange(num_jobs):
        tasks.put(Task(i, i))
    print 'they are running'
    # Add a poison pill for each consumer
#    for i in xrange(num_consumers):
#        print 'poisonning ',i
#        tasks.put(None)
#    print 'they have been poisonned'
    # Wait for all of the tasks to finish
    print 'Now waiting for them to finish'
    tasks.join()
    print 'they are finished'

    # Start printing results
    while num_jobs:
        result = results.get()
        print 'Result:', result
        num_jobs -= 1


#### simpler example trying to do the same
import multiprocessing
import time

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
                print '%s: Exiting' % proc_name
                self.task_queue.task_done()
                break
            print '%s: %s' % (proc_name, next_task)
            answer = next_task()
            self.task_queue.task_done()
            self.result_queue.put(answer)
        return

class Task(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def __call__(self):
        time.sleep(1) # pretend to take some time to do the work
        return '%s * %s = %s' % (self.a, self.b, self.a * self.b)
    def __str__(self):
        return '%s * %s' % (self.a, self.b)

tasks = multiprocessing.JoinableQueue()
results = multiprocessing.Queue()

# Start consumers
num_consumers = multiprocessing.cpu_count() * 2
print 'Creating %d consumers' % num_consumers
consumers = [ Consumer(tasks, results)
                for i in xrange(num_consumers) ]
for w in consumers:
    w.start()
    
# Enqueue jobs
num_jobs = 10
for i in xrange(num_jobs):
    tasks.put(Task(i, i))

for i in xrange(num_jobs):
    tasks.put(None)

tasks.join()

# Start printing results
while num_jobs:
    result = results.get()
    print 'Result:', result
    num_jobs -= 1













