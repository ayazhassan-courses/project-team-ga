import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt, math, networkx as nx

#Input==============================================================

def generate_input(N):
    cityList=[]
    for i in range(0,N):
        x=int(random.random()*200)
        y=int(random.random()*200)
        cityList.append((x,y))

    return cityList

#City==============================================================

def distance(citylist,city1, city2):
        xDis = abs(city1[0] - city2[0])
        yDis = abs(city1[1] - city2[1])
        distance = math.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

#Create Population==============================================================

def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route

def initialPopulation(popSize, cityList):
    population = []
    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

#Draw Route==============================================================

def drawRoute(route):
    G = nx.Graph()
    for i in range(len(route)):
        x=route[i][0]
        y=route[i][1]
        G.add_node(i,pos=(x,y))
    for n in range(1,len(route)):
        G.add_edge(n-1,n)
    pos=nx.get_node_attributes(G,'pos')
    nx.draw(G, pos, font_size=20, font_family='sans-serif')
    plt.show()

def drawGraph(graph):
    G = nx.Graph()
    for i in range(len(graph)):
        x=graph[i][0]
        y=graph[i][1]
        G.add_node(i,pos=(x,y))
    pos=nx.get_node_attributes(G,'pos')
    nx.draw(G, pos, font_size=20, font_family='sans-serif')
    plt.show()

#Determine Fitness==============================================================

def routeDistance(route):
    pathDistance=0
    for i in range(0,len(route)):
        fromCity = route[i]
        if i+1 < len(route):
            toCity=route[i+1]
        else:
            toCity=route[i]
        pathDistance += distance(route,fromCity,toCity)
    return pathDistance

def routeFitness(route):
    fitness= 1 / float(routeDistance(route))
    return fitness

def rankRoutes(population):
    fitnessResults={}
    for i in range(0,len(population)):
        fitnessResults[i]= routeFitness(population[i])
    
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)


#Select the mating pool==============================================================

def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

#Breed==============================================================

def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child

def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

#Mutate==============================================================

def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


def mutatePopulation(population, mutationRate):
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

#Repeat==============================================================

def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

#Genetic Algorithim==============================================================

def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    #Plotting
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])
    
    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]

    drawGraph(population)
    drawRoute(bestRoute)
    
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()

#Running the genetic algorithm==============================================================

#cityList = generate_input(10)
#print(cityList)

TestCitylist1 = [(96,9), (26,29), (151,122), (95,81), (177,104), (13,52), (4,73), (92,121), (129,80), (102,197)]
TestCitylist2 = [(43,172), (157,164), (190,15), (178,69), (36,70), (5,40), (8,3), (121,113), (104,112), (114,14)]

from time import time 

#Experimental Analysis
start_time = time() # record the starting time run algorithm 

geneticAlgorithm(population=TestCitylist1, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)

#No.Generations = Stopping Criteria

end_time = time() # record the ending 

time_elapsed = end_time-start_time # compute the elapsed time
print("Time taken: "+str(time_elapsed))