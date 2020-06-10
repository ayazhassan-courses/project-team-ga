import random as random


objects=[(random.randint(0, 30),random.randint(0,15))for x in range(0,10)]
knapsack_threshold=35
no_of_generations=200
#initialize population
all_possibilities=[]
for i in range(len(objects)):
    possibilities=[(random.randint(0,1))for x in range(0,len(objects))]
    all_possibilities.append(possibilities)


#fitness function
def fitness(possibility,objects,threshold):
    total_weight=0
    total_value=0
    index=0
    for i in possibility:
        if index>=len(possibility):
            break
        elif i==1:
            total_value+=objects[index][0]
            total_weight+=objects[index][1]
    if total_weight<=threshold:
            return total_weight
    else:
            return 0


#selection function
def selection(all_possibilities,objects,knapsack_threshold,num_of_parents):
    lst_of_fitness=[]
    for possibility in all_possibilities:
        fitness_values=fitness(possibility,objects,knapsack_threshold)
        lst_of_fitness.append(fitness_values)
    for i in range(num_of_parents):
        for j in range(len(lst_of_fitness)):
            if lst_of_fitness[j]==max(lst_of_fitness):
                max_fitness_idx=j
        parents= all_possibilities[max_fitness_idx]
        lst_of_fitness[max_fitness_idx] = -100000
    return parents


#crossover
def crossover(parents):
    lst_of_offsprings=[]
    crossover_point=len(parents[0])//2
    i=0
    no_of_offsprings = len(all_possibilities) - len(parents)
    while len(parents) < no_of_offsprings :
        parent1_index = i%len(parents)
        parent2_index = (i+1)%len(parents)
        offsprings=parents[parent1_index][0:crossover_point]+parents[parent2_index][crossover_point:]
        lst_of_offsprings.append(offsprings)
        i+=1
    return lst_of_offsprings 


#mutate
def mutate(children):
    mutation_rate=0.4
    index=random.randint(0,len(children)-1)
    for mutants in children:
        if mutants[index]==1:
            mutants[index]=0
        else:
            mutants[index]=1
    return mutants
    mutants =[]
    for i in range(len(children)):
        random_value = random.random()
        mutants = children[i]
        if random_value > mutation_rate:
            continue
        index=random.randint(0,len(children[i])-1)
        if mutants[index] == 0 :
            mutants[index] = 1
        else :
            mutants[index] = 0
    return mutants 


#main_function
def knap_sack_problem():
    for i in range(no_of_generations):
        fitness = fitness(possibilty,objects,knapsack_threshold)
        parents = selection(all_possibilities,objects,knapsack_threshold,5)
        offsprings = crossover(parents)
        mutants = mutation(offsprings)
        
        
