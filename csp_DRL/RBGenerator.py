import math
import random
import itertools


# Random selection from itertools.combinations(iterable, r)
def random_combination(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)


def tupleEqual(a, b):
    if (len(a)!=len(b)):
        return False
    for i in range(len(a)):
        if a[i]!=b[i]:
            return False
    return True


# Generate a forced satisfied CSP instance
def forceSat(k, n, alpha, r, p, filePath):
    file=open(filePath, 'w')
    d = int(round(pow(n, alpha)))  # dom
    m = int(round(r * n * math.log(n)))  # numConstr
    nb = int(round((1-p) * pow(d, k)))  # numAllowed
    # Write the header
    file.write("n\td\tm\tk\tnb\talpha\tr\tp\n")
    file.write(str(n)+"\t"+str(d)+"\t"+str(m)+"\t"+str(k)+"\t"+
               str(nb)+"\t"+str(alpha)+"\t"+str(r)+"\t"+str(p)+"\n")
    # Generate a random solution
    randSol=[];
    for i in range(n):
        randSol.append(random.randint(0, d-1))
    file.write(str(randSol)+"\n")
    # Generate the constraints
    for i in range(m):
        scope=random_combination(range(n), k)
        support=[tuple(randSol[j] for j in scope)]
        # Generate nb allowed tuples with repetition
        allTuples=list(itertools.product(range(d), repeat=k))
        allowedIdxes=random.sample(range(len(allTuples)), nb-1)
        for tmpIdx in allowedIdxes:
            chosenTuple=allTuples[tmpIdx];
            if tupleEqual(chosenTuple, support[0])==False:
                support.append(chosenTuple)
            else:
                replaceIdx=random.sample(set(range(len(allTuples)))-set(allowedIdxes), 1)
                support.append(allTuples[replaceIdx[0]])
        file.write(str(scope)+"|"+str(support)+"\n")


# Generate an unforced satisfied CSP instance
def unforceSat(k, n, alpha, r, p, filePath):
    file = open(filePath, 'w')
    d = int(round(pow(n, alpha)))  # dom
    m = int(round(r * n * math.log(n)))  # numConstr
    nb = int(round((1 - p) * pow(d, k)))  # numAllowed
    # Write the header
    file.write("n\td\tm\tk\tnb\talpha\tr\tp\n")
    file.write(str(n) + "\t" + str(d) + "\t" + str(m) + "\t" + str(k) + "\t" +
               str(nb) + "\t" + str(alpha) + "\t" + str(r) + "\t" + str(p) + "\n")
    # Not forced satisfied, write an empty string for solution
    file.write("" + "\n")
    # Generate the constraints
    for i in range(m):
        scope = random_combination(range(n), k)
        support = []
        # Generate nb allowed tuples without repetition
        allTuples = list(itertools.product(range(d), repeat=k))
        # allTupleIndexes = list(range(len(allTuples)))
        allowedIdxes = random.choices(list(range(len(allTuples))), k = nb - 1)
        for tmpIdx in allowedIdxes:
            support.append(allTuples[tmpIdx])
        file.write(str(scope) + "|" + str(support) + "\n")


# Generate multiple CSP instances
def genInstanceFiles(k, n, alpha, r, p, filePath, numInstances, ifSat=True):
    for i in range(numInstances):
        insFilePath=filePath+str(i)+".txt"
        if (ifSat == True):
            forceSat(k, n, alpha, r, p, insFilePath)
        if (ifSat == False):
            unforceSat(k, n, alpha, r, p, insFilePath)

# if __name__=='__main__':
#     genInstanceFiles(2, 40, 0.7, 3, 0.21, "./train_pool_randCSP./", 10)
