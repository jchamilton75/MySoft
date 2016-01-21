


#### ex 41.1
def prod(list):
    res = 1
    for x in list:
        res *= x
    return res

values = [1, 2, 3, 4]
prod(values)

#### ex 44
import collections
def adn():
    valid = set(['a', 'c', 't', 'g'])
    ok = []
    while not ok:
        answer = list(raw_input("Please type a character chain: "))
        if not answer:
            print("Empty chain")
            return 
        set_answer = set(answer)
        ok = set_answer.issubset(valid)
    print('Good ! Your chain only contains valid characters ! Now let s count them')
    #mydict = {'a': 0, 'c': 0, 't': 0, 'g': 0}
    mydict = collections.defaultdict(int)
    for elt in answer:
        mydict[elt] += 1
    print(mydict)


#### ex 46
chaine = "Ceci est une phrase sans apostrophe, qui rend le calcul du nombre de mots plus facile."
len(chaine.split())

chaine = "13 20 12.5 10 9 17"
np.mean(np.array(chaine.split()).astype(float))

def alldiff(seq):
    set_seq = set(seq)
    list_seq = list(seq)
    return len(set_seq) == len(list_seq)

nbmc=10000
sz=23
frac = 0.
for i in np.arange(nbmc):
    bla=randint(1,365,size=sz)
    if alldiff(bla): frac += 1./nbmc
print(1.-frac)
        

##### Ex 49
def permute():
    setok = set('12345')
    input = raw_input("Please type numbers 1 2 3 4 5 in any order ")
    if len(input) != len(setok):
        print("Pas le bon nombre d'elements")
        return
    setinput = set(input)
    if setinput == setok:
        print("Tous les nombres sont presents")
    else:
        print("Ce n'est pas une permutation de 1 2 3 4 5")

permute()

##### Ex 51
from CoursPython import product

def moy_geom(factors):
    n=len(factors)
    return product.prod(factors)**(1./n)

moy_geom([1,2,3,4,5,6])



##### Ex 54


