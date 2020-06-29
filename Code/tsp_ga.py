import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt, math, networkx as nx

#Input==============================================================

def generate_input(N,size): #----O(n)
    cityList=[]
    for i in range(0,N):
        x=int(random.random()*size)
        y=int(random.random()*size)
        cityList.append((x,y))

    return cityList

#City==============================================================

def distance(citylist,city1, city2): #--------------------- O(1)
        xDis = abs(city1[0] - city2[0])
        yDis = abs(city1[1] - city2[1])
        distance = math.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

#Create Population==============================================================

def createRoute(cityList): #---------------------------------------O(n)
    route = random.sample(cityList, len(cityList))
    return route

def initialPopulation(popSize, cityList): #------------------------O(m)
    population = []
    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

#Draw Route==============================================================

def drawRoute(route): #-----------------------------O(n)
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

def drawGraph(graph): #-----------------------------O(n)
    G = nx.Graph()
    for i in range(len(graph)):
        x=graph[i][0]
        y=graph[i][1]
        G.add_node(i,pos=(x,y))
    pos=nx.get_node_attributes(G,'pos')
    nx.draw(G, pos, font_size=20, font_family='sans-serif')
    plt.show()

#Determine Fitness==============================================================

def routeDistance(route): #---------------------------------O(n)
    pathDistance=0
    for i in range(0,len(route)):
        fromCity = route[i]
        if i+1 < len(route):
            toCity=route[i+1]
        else:
            toCity=route[i]
        pathDistance += distance(route,fromCity,toCity)
    return pathDistance

def routeFitness(route): #----------------------------------O(n)
    fitness= 1 / float(routeDistance(route))
    return fitness

def rankRoutes(population): #---------------------------------------------O(mn)
    fitnessResults={}
    for i in range(0,len(population)):#----m
        fitnessResults[i]= routeFitness(population[i])#---- m*n
    
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)


#Select the mating pool==============================================================

def selection(popRanked, eliteSize): #---------------------------------------------O(n*m+m^2)
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])#----- 1
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    for i in range(0, eliteSize): #------------------------------- m for both loops
        selectionResults.append(popRanked[i][0])  #---mn
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)): #---- m*m
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

def matingPool(population, selectionResults): #--------- O(n)
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

#Breed==============================================================

def breed(parent1, parent2): #--------------------------------- O(n)
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))#---n
    geneB = int(random.random() * len(parent1))#---n
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene): #-------------- n
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1] #---n

    child = childP1 + childP2
    return child

def breedPopulation(matingpool, eliteSize):#---------------------O(n)
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):  #-------------n for both loops
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

#Mutate==============================================================

def mutate(individual, mutationRate): #--------------------------O(n)
    for swapped in range(len(individual)): #-------n
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


def mutatePopulation(population, mutationRate): #------------------O(n)
    mutatedPop = []
    
    for ind in range(0, len(population)):  #---- n
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

#Repeat==============================================================

def nextGeneration(currentGen, eliteSize, mutationRate): #-------------n*m+m^2
    popRanked = rankRoutes(currentGen)#-------------------mn
    selectionResults = selection(popRanked, eliteSize) #--------------(nm+m^2)
    matingpool = matingPool(currentGen, selectionResults) #-----------n
    children = breedPopulation(matingpool, eliteSize) #--------------n
    nextGeneration = mutatePopulation(children, mutationRate)#-----n
    return nextGeneration

#Genetic Algorithim==============================================================

def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations): #----g(n*m+m^2)
    pop = initialPopulation(popSize, population) #-------------------m
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))  #-------------------mn
    #Plotting
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    g_fittest=999999
    g_bestRoute=[]
    
    #Main Loop
    for i in range(0, generations): #----------------- g
        pop = nextGeneration(pop, eliteSize, mutationRate) #-------- g(n*m+m^2)
        
        #Plot
        fittest_ind = 1 / rankRoutes(pop)[0][1]
        progress.append(fittest_ind)

        #Global Fittest
        bestRouteIndex = rankRoutes(pop)[0][0]
        bestRoute = pop[bestRouteIndex]
        fittest=routeDistance(bestRoute)

        if fittest < g_fittest:
            g_fittest=fittest
            g_bestRoute=bestRoute

        print("Generation: "+str(i)+' Distance of Fittest ind: '+str(g_fittest))
        print(bestRoute)
        print()

        

    # Result & Visualization============================

    print('==================== RESULT ========================')
    print("Best Route -> " + str(g_bestRoute))
    print("Distance ->" + str(g_fittest))
    drawGraph(population)
    drawRoute(g_bestRoute)

    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()

#Running the genetic algorithm==============================================================

cityList = generate_input(30,300)
#print(cityList)

#TestCitylist1 = [(96,9), (26,29), (151,122), (95,81), (177,104), (13,52), (4,73), (92,121), (129,80), (102,197)]
#TestCitylist2 = [(43,172), (157,164), (190,15), (178,69), (36,70), (5,40), (8,3), (121,113), (104,112), (114,14)]

from time import time 

#Experimental Analysis
start_time = time() # record the starting time run algorithm 

geneticAlgorithm(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)

#No.Generations = Stopping Criteria

end_time = time() # record the ending 

time_elapsed = end_time-start_time # compute the elapsed time
print("Time taken: "+str(time_elapsed))