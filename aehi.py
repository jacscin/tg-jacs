import matplotlib.pyplot as plt
import numpy as np
import time
from gene import Gene, mse, mape, nmse, pocid, arv, fitness, crossover

# Parameters
maxGen = 10000
mutProb = 0.1
crossProb = 0.9
popSize = 10
trainPoints = np.unique([int(i) for i in np.logspace(0, 4, 122)])
checkPoints = 10001 - trainPoints

# Load data
bbas3 = np.loadtxt('BBAS3_COTAHIST_A2010-2019.txt')
bbdc4 = np.loadtxt('BBDC4_COTAHIST_A2010-2019.txt')
itub4 = np.loadtxt('ITUB4_COTAHIST_A2010-2019.txt')
sanb11 = np.loadtxt('SANB11_COTAHIST_A2010-2019.txt')

# Run algorithm for every dataset
for data in [bbas3, bbdc4, itub4, sanb11]:
    # Show the dataset tested now
    if (data == bbas3).all():
        print("BBAS3")
    if (data == bbdc4).all():
        print("BBDC4")
    if (data == itub4).all():
        print("ITUB4")
    if (data == sanb11).all():
        print("SANB11")
    
    # Normalize data
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # Divide data into train and test
    divPoint1 = int(data.size * 0.60)
    divPoint2 = int(data.size * 0.80)
    trainData = np.array(data[:divPoint1])
    validationData = np.array(data[divPoint1:divPoint2])
    testData = np.array(data[divPoint2:])

    # Set expected values for every subset
    Ty = np.array([trainData[i] for i in range(10, trainData.size)])
    Vy = np.array([validationData[i] for i in range(10, validationData.size)])
    Ay = np.array([testData[i] for i in range(10, testData.size)])

    # Start population
    population = [Gene() for _ in range(popSize)]
    bestModel = None

    # Evolve population
    genLoss = 0
    procTrain = 0
    gen = 1
    start_time = time.time()
    while True:
        # Sometimes train a random individual
        if gen in trainPoints:
            population[np.random.randint(popSize)].fit(trainData)
        
        # Make roulette probabilities
        population.sort(key=lambda ind: ind.getFitness(trainData), reverse=True)
        rouletteProbs = np.array([i.getFitness(trainData) for i in population])
        rouletteProbs = rouletteProbs / np.sum(rouletteProbs)

        # Choose parents
        poolProbs = rouletteProbs[:5]/np.sum(rouletteProbs[:5])
        parents = np.random.choice(population[:5], 2, False, poolProbs)

        # Remove parents from population
        population = np.delete(population, [i for i in range(len(population)) if population[i] in parents])

        # Update rouletteProbs for new population
        rouletteProbs = np.array([i.getFitness(trainData) for i in population])
        rouletteProbs = rouletteProbs / np.sum(rouletteProbs)

        # Crossover
        offspring = crossover(parents, crossProb, mutProb, trainData)

        # Make new population
        if len(population) >= popSize - len(parents) - len(offspring):
            population = list(parents) + list(offspring) + list(np.random.choice(population, popSize - len(parents) - len(offspring), False, rouletteProbs))
        else:
            population = list(parents) + list(offspring) + list(population) + [Gene() for _ in range(popSize - len(parents) - len(offspring) - len(population))]

        # Get performance
        fits = [i.getFitness(trainData) for i in population]
        bestModel = population[np.argmax(fits)]
        
        # Stop criteria
        if gen in checkPoints:
            # Get metrics of best model
            trainFit = bestModel.getFitness(trainData)
            validFit = bestModel.getFitness(validationData)
            if ((trainFit >= 40) or
            ((genLoss - validFit) > (0.05 * genLoss)) or
            ((abs(procTrain - trainFit) <= 1e-6) and (abs(procTrain - trainFit) != 0)) or
            (gen >= maxGen)):
                break
            else:
                genLoss = validFit
                procTrain = trainFit

        gen += 1

    # Print execution time
    print("--- %s seconds ---" % (time.time() - start_time))

    # Get best model
    print("netMod:", bestModel.netMod)
    print("nHidden:", bestModel.nHidden)
    print("nLags:", bestModel.nLags)
    print("nLags:", bestModel.lags)

    # Get metrics of best model
    Py = bestModel.predict(testData)
    print("MSE:", mse(Ay, Py))
    print("MAPE:", mape(Ay, Py))
    print("NMSE:", nmse(Ay, Py))
    print("POCID:", pocid(Ay, Py))
    print("ARV:", arv(Ay, Py))
    print("Fitness:", bestModel.getFitness(testData))

""" # Plot results
xaxis = range(1, testData.size - 10 + 1)
plt.plot(xaxis, Ay)
plt.plot(xaxis, Py)
plt.annotate('Fit = {:.6f}'.format(fitness(Ay, Py)), xy=(0.914, 0.9), xycoords='axes fraction')
plt.legend(["Actual values", "Predicted values"])
plt.show() """