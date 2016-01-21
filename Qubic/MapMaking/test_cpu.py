import sys
import numpy as np
import time
    
def doit(nnn):
    t0=time.time()
    matrnd=np.random.uniform(size=(nnn,nnn))
    t1=time.time()
    vecrnd=np.random.uniform(size=nnn)
    t2=time.time()
    prodrnd=np.dot(matrnd,vecrnd)
    t3=time.time()
    prodmat=np.dot(matrnd,matrnd)
    t4=time.time()
    return t1-t0,t2-t1,t3-t2,t4-t3


def main(argv):
    nnn=int(sys.argv[1])
    print('N='+np.str(nnn))
    t0,t1,t2,t3=doit(nnn)
    print('Allocation matrice: {0:.4f} sec.'.format(t0))
    print('Allocation vecteur: {0:.4f} sec.'.format(t1))
    print('Produit matrice.vecteur: {0:.4f} sec.'.format(t2))
    print('Produit matrice.matrice: {0:.4f} sec.'.format(t3))


if __name__ == "__main__":
    main(sys.argv)
