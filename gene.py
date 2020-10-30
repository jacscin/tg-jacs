import numpy as np

def mse(Ay, Py):
    # Lower is better
    return np.mean((Ay - Py) ** 2)

def mape(Ay, Py):
    # Lower is better
    Ay[np.where(Ay == 0)] = Py[np.where(Ay == 0)] / (1 - Py[np.where(Ay == 0)])
    return (100 * np.mean(np.abs((Ay - Py) / Ay)))

def nmse(Ay, Py):
    # Lower is better
    return (np.sum((Ay - Py) ** 2) / np.sum(np.diff(Py) ** 2))

def pocid(Ay, Py):
    # Higher is better
    return (100 * np.mean(list(map(int, np.diff(Ay)*np.diff(Py) > 0))))

def arv(Ay, Py):
    # Lower is better
    return (np.sum((Py - Ay) ** 2) / np.sum((Py - np.mean(Ay)) ** 2))

def fitness(Ay, Py):
    return (pocid(Ay, Py) / (1 + mse(Ay, Py) + mape(Ay, Py) + nmse(Ay, Py) + arv(Ay, Py)))

def crossover(parents, w, y, data):
    offspring = [Gene() for _ in range(4)]
    # Making C1
    # Calc netMod
    offspring[0].netMod = int((parents[0].netMod + parents[1].netMod) / 2)
    # Calc nHidden
    offspring[0].nHidden = int((parents[0].nHidden + parents[1].nHidden) / 2)
    # Calc nLags
    offspring[0].nLags = int((parents[0].nLags + parents[1].nLags) / 2)
    # Calc lags
    lim = min(parents[0].lags.size, parents[1].lags.size)
    offspring[0].lags = (parents[0].lags[:lim] + parents[1].lags[:lim]) / 2
    offspring[0].lags = np.concatenate((offspring[0].lags, parents[0].lags[lim:],  parents[1].lags[lim:]))
    offspring[0].lags = np.unique(offspring[0].lags.astype(int))
    offspring[0].lags = offspring[0].lags[:offspring[0].nLags]
    while(offspring[0].lags.size < offspring[0].nLags):
        n = [i for i in range(10) if i not in offspring[0].lags]
        offspring[0].lags = np.append(offspring[0].lags, np.random.choice(n))
    offspring[0].lags = np.sort(offspring[0].lags)
    # Calc wji
    shp1 = parents[0].wji.shape
    shp2 = parents[1].wji.shape
    if shp1 == shp2:
        offspring[0].wji = (parents[0].wji + parents[1].wji) / 2
    else:
        shp = (max(shp1[0], shp2[0]), max(shp1[1], shp2[1]))
        offspring[0].wji = np.pad(parents[0].wji, ((0, shp[0] - shp1[0]), (0, shp[1] - shp1[1])), 'constant')
        offspring[0].wji = offspring[0].wji + np.pad(parents[1].wji, ((0, shp[0] - shp2[0]), (0, shp[1] - shp2[1])), 'constant')
        offspring[0].wji = offspring[0].wji / 2
        offspring[0].wji = np.atleast_2d(offspring[0].wji[:offspring[0].nHidden, :offspring[0].nLags])
    # Calc wkj
    lim = min(parents[0].wkj.size, parents[1].wkj.size)
    offspring[0].wkj = (parents[0].wkj[0,:lim] + parents[1].wkj[0,:lim]) / 2
    offspring[0].wkj = np.concatenate((offspring[0].wkj, parents[0].wkj[0,lim:],  parents[1].wkj[0,lim:]))
    offspring[0].wkj = np.atleast_2d(offspring[0].wkj[:offspring[0].nHidden])
    offspring[0].nHidden = offspring[0].wkj.size
    # Calc bj
    lim = min(parents[0].bj.size, parents[1].bj.size)
    offspring[0].bj = (parents[0].bj[:lim] + parents[1].bj[:lim]) / 2
    offspring[0].bj = np.concatenate((offspring[0].bj, parents[0].bj[lim:],  parents[1].bj[lim:]))
    offspring[0].bj = offspring[0].bj[:offspring[0].nHidden]
    offspring[0].nHidden = offspring[0].bj.size
    # Calc bk
    offspring[0].bk = (parents[0].bk + parents[1].bk) / 2

    # Making C2
    # Calc netMod
    offspring[1].netMod = max(parents[0].netMod, parents[1].netMod)
    # Calc nHidden
    offspring[1].nHidden = max(parents[0].nHidden, parents[1].nHidden)
    # Calc nLags
    offspring[1].nLags = max(parents[0].nLags, parents[1].nLags)
    # Calc lags
    lim = min(parents[0].lags.size, parents[1].lags.size)
    pmax = max(np.max(parents[0].lags), np.max(parents[1].lags))
    offspring[1].lags = np.maximum(parents[0].lags[:lim], parents[1].lags[:lim])
    offspring[1].lags = np.concatenate((offspring[1].lags, parents[0].lags[lim:], parents[1].lags[lim:]))
    offspring[1].lags = (offspring[1].lags * w) + ((1 - w) * pmax)
    offspring[1].lags = np.unique(offspring[1].lags.astype(int))
    offspring[1].lags = offspring[1].lags[:offspring[1].nLags]
    while(offspring[1].lags.size < offspring[1].nLags):
        n = [i for i in range(10) if i not in offspring[1].lags]
        offspring[1].lags = np.append(offspring[1].lags, np.random.choice(n))
    offspring[1].lags = np.sort(offspring[1].lags)
    # Calc wji
    shp1 = parents[0].wji.shape
    shp2 = parents[1].wji.shape
    pmax = max(np.max(parents[0].wji), np.max(parents[1].wji))
    if shp1 == shp2:
        offspring[1].wji = (np.maximum(parents[0].wji, parents[1].wji) * w) + ((1 - w) * pmax)
    else:
        shp = (max(shp1[0], shp2[0]), max(shp1[1], shp2[1]))
        offspring[1].wji = np.zeros(shp)
        offspring[1].wji += (pmax * (1 - w))
        offspring[1].wji += (w * np.maximum(np.pad(parents[0].wji, ((0, shp[0] - shp1[0]), (0, shp[1] - shp1[1])), 'constant'),
                                            np.pad(parents[1].wji, ((0, shp[0] - shp2[0]), (0, shp[1] - shp2[1])), 'constant')))
        offspring[1].wji = np.atleast_2d(offspring[1].wji[:offspring[1].nHidden, :offspring[1].nLags])
    # Calc wkj
    lim = min(parents[0].wkj.size, parents[1].wkj.size)
    pmax = max(np.max(parents[0].wkj), np.max(parents[1].wkj))
    offspring[1].wkj = np.maximum(parents[0].wkj[0,:lim], parents[1].wkj[0,:lim])
    offspring[1].wkj = np.concatenate((offspring[1].wkj, parents[0].wkj[0,lim:], parents[1].wkj[0,lim:]))
    offspring[1].wkj = offspring[1].wkj[:offspring[1].nHidden]
    offspring[1].wkj = np.atleast_2d((offspring[1].wkj * w) + (pmax * (1 - w)))
    offspring[1].nHidden = offspring[1].wkj.size
    # Calc bj
    lim = min(parents[0].bj.size, parents[1].bj.size)
    pmax = max(np.max(parents[0].bj), np.max(parents[1].bj))
    offspring[1].bj = np.maximum(parents[0].bj[:lim], parents[1].bj[:lim])
    offspring[1].bj = np.concatenate((offspring[1].bj, parents[0].bj[lim:],  parents[1].bj[lim:]))
    offspring[1].bj = offspring[1].bj[:offspring[1].nHidden]
    offspring[1].bj = (offspring[1].bj * w) + (pmax * (1 - w))
    offspring[1].nHidden = offspring[1].bj.size
    # Calc bk
    offspring[1].bk = max(parents[0].bk, parents[1].bk)

    # Making C3
    # Calc netMod
    offspring[2].netMod = min(parents[0].netMod, parents[1].netMod)
    # Calc nHidden
    offspring[2].nHidden = min(parents[0].nHidden, parents[1].nHidden)
    # Calc nLags
    offspring[2].nLags = min(parents[0].nLags, parents[1].nLags)
    # Calc lags
    lim = min(parents[0].lags.size, parents[1].lags.size)
    pmin = min(np.min(parents[0].lags), np.min(parents[1].lags))
    offspring[2].lags = np.minimum(parents[0].lags[:lim], parents[1].lags[:lim])
    offspring[2].lags = np.concatenate((offspring[2].lags, parents[0].lags[lim:],  parents[1].lags[lim:]))
    offspring[2].lags = (offspring[2].lags * w) + ((1 - w) * pmin)
    offspring[2].lags = np.unique(offspring[2].lags.astype(int))
    offspring[2].lags = offspring[2].lags[:offspring[2].nLags]
    offspring[2].lags = np.sort(offspring[2].lags)
    while(offspring[2].lags.size < offspring[2].nLags):
        n = [i for i in range(10) if i not in offspring[2].lags]
        offspring[2].lags = np.append(offspring[2].lags, np.random.choice(n))
    offspring[2].lags = np.sort(offspring[2].lags)
    # Calc wji
    shp1 = parents[0].wji.shape
    shp2 = parents[1].wji.shape
    pmin = min(np.min(parents[0].wji), np.min(parents[1].wji))
    if shp1 == shp2:
        offspring[2].wji = (np.minimum(parents[0].wji, parents[1].wji) * w) + ((1 - w) * pmin)
    else:
        shp = (max(shp1[0], shp2[0]), max(shp1[1], shp2[1]))
        offspring[2].wji = np.zeros(shp)
        offspring[2].wji += (pmin * (1 - w))
        offspring[2].wji += (w * np.minimum(np.pad(parents[0].wji, ((0, shp[0] - shp1[0]), (0, shp[1] - shp1[1])), 'constant'),
                                            np.pad(parents[1].wji, ((0, shp[0] - shp2[0]), (0, shp[1] - shp2[1])), 'constant')))
        offspring[2].wji = np.atleast_2d(offspring[2].wji[:offspring[2].nHidden, :offspring[2].nLags])
    # Calc wkj
    lim = min(parents[0].wkj.size, parents[1].wkj.size)
    pmin = min(np.min(parents[0].wkj), np.min(parents[1].wkj))
    offspring[2].wkj = np.minimum(parents[0].wkj[0,:lim], parents[1].wkj[0,:lim])
    offspring[2].wkj = np.concatenate((offspring[2].wkj, parents[0].wkj[0,lim:], parents[1].wkj[0,lim:]))
    offspring[2].wkj = offspring[2].wkj[:offspring[2].nHidden]
    offspring[2].wkj = np.atleast_2d((offspring[2].wkj * w) + (pmin * (1 - w)))
    offspring[2].nHidden = offspring[2].wkj.size
    # Calc bj
    lim = min(parents[0].bj.size, parents[1].bj.size)
    pmin = min(np.min(parents[0].bj), np.min(parents[1].bj))
    offspring[2].bj = np.minimum(parents[0].bj[:lim], parents[1].bj[:lim])
    offspring[2].bj = np.concatenate((offspring[2].bj, parents[0].bj[lim:],  parents[1].bj[lim:]))
    offspring[2].bj = offspring[2].bj[:offspring[2].nHidden]
    offspring[2].bj = (offspring[2].bj * w) + (pmin * (1 - w))
    offspring[2].nHidden = offspring[2].bj.size
    # Calc bk
    offspring[2].bk = min(parents[0].bk, parents[1].bk)
    
    # Making C4
    # Calc netMod
    offspring[3].netMod = int((parents[0].netMod + parents[1].netMod) / 2)
    # Calc nHidden
    offspring[3].nHidden = int((parents[0].nHidden + parents[1].nHidden) / 2)
    # Calc nLags
    offspring[3].nLags = int((parents[0].nLags + parents[1].nLags) / 2)
    # Calc lags
    lim = min(parents[0].lags.size, parents[1].lags.size)
    pmin = min(np.min(parents[0].lags), np.min(parents[1].lags))
    pmax = max(np.max(parents[0].lags), np.max(parents[1].lags))
    offspring[3].lags = parents[0].lags[:lim] + parents[1].lags[:lim]
    offspring[3].lags = np.concatenate((offspring[3].lags, parents[0].lags[lim:], parents[1].lags[lim:]))
    offspring[3].lags = ((offspring[3].lags * w) + ((1 - w) * (pmax + pmin))) / 2
    offspring[3].lags = np.unique(offspring[3].lags.astype(int))
    offspring[3].lags = offspring[3].lags[:offspring[3].nLags]
    offspring[3].lags = np.sort(offspring[3].lags)
    while(offspring[3].lags.size < offspring[3].nLags):
        n = [i for i in range(10) if i not in offspring[3].lags]
        offspring[3].lags = np.append(offspring[3].lags, np.random.choice(n))
    offspring[3].lags = np.sort(offspring[3].lags)
    # Calc wji
    shp1 = parents[0].wji.shape
    shp2 = parents[1].wji.shape
    pmin = min(np.min(parents[0].wji), np.min(parents[1].wji))
    pmax = min(np.max(parents[0].wji), np.max(parents[1].wji))
    if shp1 == shp2:
        offspring[3].wji = (((parents[0].wji + parents[1].wji) * w) + ((1 - w) * pmin)) / 2
    else:
        shp = (max(shp1[0], shp2[0]), max(shp1[1], shp2[1]))
        offspring[3].wji = np.zeros(shp)
        offspring[3].wji += ((pmax + pmin) * (1 - w))
        offspring[3].wji += (w * (np.pad(parents[0].wji, ((0, shp[0] - shp1[0]), (0, shp[1] - shp1[1])), 'constant') +
                                  np.pad(parents[1].wji, ((0, shp[0] - shp2[0]), (0, shp[1] - shp2[1])), 'constant')))
        offspring[3].wji = np.atleast_2d(offspring[3].wji[:offspring[3].nHidden, :offspring[3].nLags] / 2)
    # Calc wkj
    lim = min(parents[0].wkj.size, parents[1].wkj.size)
    pmin = min(np.min(parents[0].wkj), np.min(parents[1].wkj))
    pmax = max(np.max(parents[0].wkj), np.max(parents[1].wkj))
    offspring[3].wkj = parents[0].wkj[0,:lim] + parents[1].wkj[0,:lim]
    offspring[3].wkj = np.concatenate((offspring[3].wkj, parents[0].wkj[0,lim:], parents[1].wkj[0,lim:]))
    offspring[3].wkj = offspring[3].wkj[:offspring[3].nHidden]
    offspring[3].wkj = np.atleast_2d(((offspring[3].wkj * w) + ((pmax + pmin) * (1 - w))) / 2)
    offspring[3].nHidden = offspring[3].wkj.size
    # Calc bj
    lim = min(parents[0].bj.size, parents[1].bj.size)
    pmin = min(np.min(parents[0].bj), np.min(parents[1].bj))
    pmax = max(np.min(parents[0].bj), np.max(parents[1].bj))
    offspring[3].bj = parents[0].bj[:lim] + parents[1].bj[:lim]
    offspring[3].bj = np.concatenate((offspring[3].bj, parents[0].bj[lim:],  parents[1].bj[lim:]))
    offspring[3].bj = offspring[3].bj[:offspring[3].nHidden]
    offspring[3].bj = ((offspring[3].bj * w) + ((pmax + pmin) * (1 - w))) / 2
    offspring[3].nHidden = offspring[3].bj.size
    # Calc bk
    offspring[3].bk = (parents[0].bk + parents[1].bk) / 2

    # Find cbest
    offspring.sort(key=lambda cr: cr.getFitness(data), reverse=True)
    cbest = offspring[0]
    Mk = [cbest, cbest, cbest]

    # Generate Yk
    Y1 = np.zeros(8)
    Y1[np.random.randint(8)] = 1
    Y2 = np.zeros(8)
    Y2[np.random.choice(8, np.random.randint(8), replace=False)] = 1
    Y3 = np.ones(8)
    Yk = [Y1, Y2, Y3]

    # Mutate Mjs
    for M, Y in zip(Mk, Yk):
        if np.random.rand() < y:
            if Y[0] == 1:
                pmin = min(parents[0].netMod, parents[1].netMod)
                pmax = max(parents[0].netMod, parents[1].netMod)
                dMk = (np.random.rand() * (pmax - pmin)) + pmin - M.netMod
                M.netMod = M.netMod + int(dMk)
            
            if Y[1] == 1:
                pmin = min(parents[0].nHidden, parents[1].nHidden)
                pmax = max(parents[0].nHidden, parents[1].nHidden)
                dMk = (np.random.rand() * (pmax - pmin)) + pmin - M.nHidden
                M.nHidden = M.nHidden + int(dMk)
            
            if Y[2] == 1:
                pmin = min(parents[0].nLags, parents[1].nLags)
                pmax = max(parents[0].nLags, parents[1].nLags)
                dMk = (np.random.rand() * (pmax - pmin)) + pmin - M.nLags
                M.nLags = M.nLags + int(dMk)
            
            if Y[3] == 1:
                pmin = min(np.min(parents[0].lags), np.min(parents[1].lags))
                pmax = max(np.max(parents[0].lags), np.max(parents[1].lags))
                dMk = (np.random.random_sample(M.lags.shape) * (pmax - pmin)) + pmin - M.lags
                M.lags = M.lags + dMk.astype(int)
                M.lags = np.unique(M.lags)
                M.lags = np.sort(M.lags)
                while(M.lags.size < M.nLags):
                    n = [i for i in range(10) if i not in M.lags]
                    M.lags = np.append(M.lags, np.random.choice(n))
                if (M.lags.size > M.nLags):
                    M.lags = M.lags[:M.nLags]

            if Y[4] == 1:
                pmin = min(np.min(parents[0].wji), np.min(parents[1].wji))
                pmax = max(np.max(parents[0].wji), np.max(parents[1].wji))
                dMk = (np.random.random_sample(M.wji.shape) * (pmax - pmin)) + pmin - M.wji
                M.wji = M.wji + dMk
                shp = M.wji.shape
                if (shp[0] > M.nHidden):
                    M.wji = M.wji[:M.nHidden,:]
                elif (shp[0] < M.nHidden):
                    M.wji = np.pad(M.wji, ((0, M.nHidden - shp[0]), (0, 0)), 'constant')
                    M.wji[shp[0]:,:] = (np.random.rand(M.nHidden - shp[0], shp[1]) * 2 * 0.3) - 0.3
                if (shp[1] > M.nLags):
                    M.wji = M.wji[:,:M.nLags]
                elif (shp[1] < M.nLags):
                    M.wji = np.pad(M.wji, ((0, 0), (0, M.nLags - shp[1])), 'constant')
                    M.wji[:,shp[1]:] = (np.random.rand(M.wji.shape[0], M.nLags - shp[1]) * 2 * 0.3) - 0.3

            if Y[5] == 1:
                pmin = min(np.min(parents[0].wkj), np.min(parents[1].wkj))
                pmax = max(np.max(parents[0].wkj), np.max(parents[1].wkj))
                dMk = (np.random.random_sample(M.wkj.shape) * (pmax - pmin)) + pmin - M.wkj
                M.wkj = M.wkj + dMk
                shp = M.wkj.size
                if (shp > M.nHidden):
                    M.wkj = np.atleast_2d(M.wkj[0,:M.nHidden])
                elif (shp < M.nHidden):
                    tail = (np.random.rand(1, M.nHidden - shp) * 2 * 0.3) - 0.3
                    M.wkj = np.concatenate((M.wkj, tail), axis=1)
            
            if Y[6] == 1:
                pmin = min(np.min(parents[0].bj), np.min(parents[1].bj))
                pmax = max(np.max(parents[0].bj), np.max(parents[1].bj))
                dMk = (np.random.random_sample(M.bj.shape) * (pmax - pmin)) + pmin - M.bj
                M.bj = M.bj + dMk
            
            if Y[7] == 1:
                pmin = min(parents[0].bk, parents[1].bk)
                pmax = max(parents[0].bk, parents[1].bk)
                dMk = (np.random.rand() * (pmax - pmin)) + pmin - M.bk
                M.bk = M.bk + int(dMk)

    

    """ # Show bug
    for ind in Mk:
        if (ind.lags.size != ind.nLags):
            print(ind.lags)
            print(ind.nLags)
            print(ind.wji.shape) """

    return Mk

class Gene:
    def __init__(self):
        self.netMod = np.random.randint(1,4)
        self.nHidden = np.random.randint(1,11)
        self.nLags = np.random.randint(1,10)
        self.lags = np.sort(np.random.choice(10, self.nLags, False))
        
        eps = 0.3
        self.wji = (np.random.rand(self.nHidden, self.nLags) * 2 * eps) - eps # 10xnLags
        self.wkj = (np.random.rand(1, self.nHidden) * 2 * eps) - eps # 1x10
        self.bj = (np.random.rand(self.nHidden, 1) * 2 * eps) - eps # 10x1
        self.bk = (np.random.rand(1, 1) * 2 * eps) - eps # 1x1
    
    def sig(self, x):
        return (1/(1 + np.exp(-x)))

    def sigC(self, x):
        return (self.sig(x) * (1 - self.sig(x)))

    def predict(self, data):
        Ax = np.array([data[i+self.lags] for i in range(data.size - 10)])
        Py = 0
        z1 = (self.wji @ Ax.T) + self.bj # (10xnLags * nLagsxnSamples) + 10x1 = 10xnSamples
        a1 = self.sig(z1) # 10xnSamples
        z2 = self.wkj @ a1 # (1x10 * 10xnSamples) = 1xnSamples

        if self.netMod == 1:
            z2 = z2 + self.sig(self.bk) # 1xnSamples + 1x1 = 1xnSamples
        else:
            z2 = z2 + self.bk # 1xnSamples + 1x1 = 1xnSamples

        if self.netMod == 3:
            Py = self.sig(z2) # 1xnSamples
        else:
            Py = z2 # 1xnSamples

        return Py.ravel()
    
    def fit(self, data):
        Ax = np.array([data[i+self.lags] for i in range(data.size - 10)])
        Ay = np.array([data[i] for i in range(10, data.size)])
        epochs = 0
        while epochs < 300:
            wjiGrad = np.zeros(self.wji.shape)
            wkjGrad = np.zeros(self.wkj.shape)
            bjGrad = np.zeros(self.bj.shape)
            bkGrad = np.zeros(self.bk.shape)
            for i in range(Ax.shape[0]): # nSamples iterations
                Py = 0
                z1 = (self.wji @ np.atleast_2d(Ax[i,:]).T) + self.bj # (10xnLags * nLagsx1) + 10x1 = 10x1
                a1 = self.sig(z1) # 10x1
                z2 = self.wkj @ a1 # (1x10 * 10x1) = 1x1

                if self.netMod == 1:
                    z2 = z2 + self.sig(self.bk) # 1x1 + 1x1 = 1x1
                else:
                    z2 = z2 + self.bk # 1x1 + 1x1 = 1x1

                if self.netMod == 3:
                    Py = self.sig(z2) # 1x1
                else:
                    Py = z2 # 1x1
                
                d2 = Py - Ay[i] # 1x1
                if self.netMod == 3:
                    d2 = self.sigC(z2) * d2
                d2 = self.sigC(z2) * d2
                d1 = (self.wkj.T @ d2) * self.sigC(z1) # (10x1 * 1x1) * 10x1 = 10x1
                
                wjiGrad += (d1 @ np.atleast_2d(Ax[i,:])) # 10xnLags + (10x1 * 1xnLags) = 10xnLags
                wkjGrad += (d2 * a1.T) # 1x10 + (1x1 * 1x10) = 1x10
                bjGrad += d1
                bkGrad += d2

            # Update
            n = 1
            wjiGrad = n * wjiGrad/Ax.shape[0] #+ ((0.1/Ax.shape[0]) * self.wji)
            wkjGrad = n * wkjGrad/Ax.shape[0] #+ ((0.1/Ax.shape[0]) * self.wkj)
            bjGrad = n * bjGrad/Ax.shape[0]
            bkGrad = n * bkGrad/Ax.shape[0]
            self.wji -= wjiGrad
            self.wkj -= wkjGrad
            self.bj -= bjGrad
            self.bk -= bkGrad

            # Check performance
            err = mse(Ay, self.predict(data))
            if (err < 0.001):
                break
            epochs = epochs + 1
    
    def getFitness(self, data):
        Ay = np.array([data[i] for i in range(10, data.size)])
        try:
            Py = self.predict(data)
        except:
            return 0
        return fitness(Ay, Py)