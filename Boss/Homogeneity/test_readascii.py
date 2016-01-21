

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def loadRand(fin,qsize=4,rbins=50,randsize=10):
    ''' 
    takes a string name of a file with data fin
    (qsize x rbins) = ( number of columns ) x ( number of rows )
    returns an array with randomly 
    sampling this file, with randsize-size 
    '''
    print 'start of producing rand indices'
    # one way to produce rand indices
    #aaa = random.sample(range(rbins),randsize)

    # JC efficient way   
    	### there is a bug here: the size should be number of lines in the file
  	randindices = np.random.rand(rbins)
    xx = np.argsort(randindices)
    aaa= xx[:randsize]
    
    res = np.zeros([np.shape(aaa)[0],qsize])
    print 'start loading file'
    i=0
    with open(fin) as fd:
        print 'files is open!'
        for n, line in enumerate(fd):
            if (n in aaa):
                res[i] = np.fromstring(line.strip(),sep=' ')
                print i,n,res[i]
                i = i + 1
    
    return res



def loadRandNew(fin,qsize=4,rbins=50,randsize=10):
	nblines = file_len(fin)

	randindices = np.random.rand(nblines)
	xx = np.argsort(randindices)
	aaa= xx[:randsize]
	#print(aaa)

	res = np.zeros([np.shape(aaa)[0],qsize])
	#print 'start loading file'
	i=0
	with open(fin) as fd:
		#print 'files is open!'
		for n, line in enumerate(fd):
			if (n in aaa):
				res[i] = np.fromstring(line.strip(),sep=' ')
				#print i,n,res[i]
				i = i + 1

	return res

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1




def loadRandJC(fin,qsize=4,randsize=10):
	nblines = file_len(fin)

	randindices = np.random.rand(nblines)
	xx = np.argsort(randindices)
	aaa= np.sort(xx[:randsize])
	#print(aaa)
	res = np.zeros([np.shape(aaa)[0],qsize])
	#print 'start loading file'
	i=0
	with open(fin) as fd:
		#print 'files is open!'
		for n, line in enumerate(fd):
			if n == aaa[i]:
				res[i] = np.fromstring(line.strip(),sep=' ')
				#print i,n,res[i]
				i = i + 1
				if i==randsize:
					break


	return res

bla = loadRandJC('toto', randsize=10)


bla = loadRandJC('a0.6452_rand50x_v2.dr12d_cmass_ngc.rdz', randsize=10)


#%timeit bla = loadRand('toto', randsize=10)
%timeit bla = loadRandNew('toto', randsize=10)
%timeit bla = loadRandJC('toto', randsize=10)






