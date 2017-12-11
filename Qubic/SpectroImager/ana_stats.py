


######## Old

f = open('/Users/hamilton/CMB/Interfero/SpectroImager/logs_tycho.txt', 'r') # 'r' = read
lines = f.readlines()
f.close()

nj = []
tt = []
for l in lines:
	bla = l.split('_')
	blo = l.split(' ')
	nj.append(bla[3])
	tt.append(blo[4])

nj = np.array(nj)
tt = np.array(tt)
nodes = np.array([1,2,4,1,2,4,4,1,2,4,1,2,4,1,2])
tasks = np.array([4,4,4,8,8,8,16,2,2,2,1,1,1,16,16])
threads = np.array([4,4,4,2,2,2,1,8,8,8,16,16,16,1,1])


valnodes = np.unique(nodes)
valthreads = np.unique(threads)
valtasks = np.unique(tasks)

clf()
for n in valnodes:
	mm = nodes==n
	thetasks = tasks[mm]
	thetime = tt[mm]
	plot(thetasks[argsort(thetasks)], thetime[argsort(thetasks)], label='Nnodes = {}'.format(n))
legend()
xlabel('Tasks per Node')



###### New
jobid, elapsed, vmem, nnodes, ntasks, nthreads = np.loadtxt('logs_tycho2.txt').T
valnodes = np.unique(nnodes)
valthreads = np.unique(nthreads)
valtasks = np.unique(ntasks)

clf()
for n in valnodes:
	mm = nnodes==n
	thetasks = ntasks[mm]
	thetime = elapsed[mm]
	plot(thetasks[argsort(thetasks)], thetime[argsort(thetasks)], lw=3, label='Nnodes = {}'.format(n))
legend()
xlabel('Tasks per Node')
ylabel('Seconds')
xlim(0,17)


clf()
for n in valnodes:
	mm = nnodes==n
	thetasks = ntasks[mm]
	themem = vmem[mm]/1e6
	aa = argsort(thetasks)
	thetasks = thetasks[aa]
	themem = themem[aa]
	plot(thetasks, themem, lw=3, label='Nnodes = {}'.format(n))
legend(loc='upper left')
xlabel('Tasks per Node')
ylabel('Mem Gb')
xlim(0,17)






