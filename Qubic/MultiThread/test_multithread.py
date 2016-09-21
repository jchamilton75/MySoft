
################ La ça marche !!!!!!
from MultiThread import multi_process as multi
import time

def test(i,arguments, common_arguments):
    a,b,c = common_arguments
    d = arguments
    output =[]
    for args in d:
        time.sleep(np.random.rand()/a)
        output.append(args**2)
    return(output)

individual_arguments = np.arange(8)
common_arguments = [10,'bb','cc']
out = multi.parallel_for(test, individual_arguments, common_arguments,nprocs=8)
######################


### Tests:
individual_arguments = np.arange(1000)
common_arguments = [100,'bb','cc']
nprocs = arange(16)+1
alltimes=[]
for n in nprocs:
    t = %timeit -o multi.parallel_for(test, individual_arguments, common_arguments,nprocs=n)
    alltimes.append(t)
    
thetimes = [5.47, 2.82, 1.9, 1.44, 1.18, 1.03, 0.896, 0.752, 0.681, 0.616, 0.614, 0.539, 0.545, 0.490, 0.482, 0.451]

plot(nprocs, thetimes, 'ro')
yscale('log')
xscale('log')
xlim(0.9, 20)
plot(nprocs, 5.5/nprocs)




#### Exemple nico: somme des n premiers entiers
def test_nico(i,arguments, common_arguments):
    d = arguments
    output =[]
    s=0
    for args in d:
        s = s+d
        output.append(s)
    return(s)

individual_arguments = np.arange(100)
common_arguments = []
out = multi.parallel_for(test_nico, individual_arguments, common_arguments,nprocs=10)

#### A comprendre





