from tabulate import tabulate
grille = [	[0, 0, 1, 0, 5, 4, 0, 0, 0],
			[7, 5, 0, 6, 0, 0, 8, 0, 0],
			[8, 0, 4, 0, 0, 0, 3, 0, 2],
			[0, 0, 7, 0, 0, 1, 5, 0, 3],
			[9, 8, 0, 2, 0, 7, 0, 4, 6],
			[1, 0, 6, 5, 0, 0, 7, 0, 0],
			[5, 0, 2, 0, 0, 0, 4, 0, 8],
			[0, 0, 3, 0, 0, 5, 0, 9, 1],
			[0, 0, 0, 7, 1, 0, 6, 0, 0]]


grille = np.reshape(grille, (3,3,3,3))

def print_grille(grille):




def lignes(g):
	ll = []
	for i in xrange(3):
		for j in xrange(3):
			ll.append(np.ravel(g[i,j,:,:]))
	return ll


def colonnes(g):
	cc = []
	for i in xrange(3):
		for j in xrange(3):
			cc.append(np.ravel(g[:,:,i,j]))
	return cc

def carre(g):
	cc = []
	for i in xrange(3):
		for j in xrange(3):
			cc.append(np.ravel(g[i,:,j,:]))
	return cc


def inconnus(grille):
	inc = grille == 0
	return inc
















