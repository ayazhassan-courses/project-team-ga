# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 21:58:59 2020

@author: Administrator
"""


import random
from mpl_toolkits import mplot3d
import math
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(projection='3d')
################################# CODE FOR THE 3D  PLOT ##########################


def math_function(x, y):
    return -((x-3) ** 2 + y ** 2)+5

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = math_function(X, Y)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
##############################################################################33

#Generate random solutions {x,y} within the domain of the function 
#and adds them to population
def generate_population(population_size, x_domain, y_domain):
    lower_x_boundary, upper_x_boundary = x_domain
    lower_y_boundary, upper_y_boundary = y_domain
    
    population = []
    for i in range(population_size):
        individual = {
            "x": round(random.uniform(lower_x_boundary, upper_x_boundary),3),
            "y": round(random.uniform(lower_y_boundary, upper_y_boundary),3)
        }
        population.append(individual)

    return population

################ OBJECTIVE FUNCTION WE WANT TO MAXIMIZE ##################
def apply_fitness_function(individual):
    x = individual["x"]
    y = individual["y"]
    return round((-((x-3) ** 2 + y ** 2)+5),3)

################## SELECTION STRATEGY TO FORM A PINWHEEL AND SELECT A RANDOM NUMBER
def choice_by_roulette(sorted_population, fitness_sum):
    offset = 0
    normalized_fitness_sum = fitness_sum #total fitness

    lowest_fitness = apply_fitness_function(sorted_population[0])
    if lowest_fitness < 0:
        offset = -lowest_fitness
        normalized_fitness_sum += offset * len(sorted_population)

    draw = random.uniform(0, 1) #random number

    accumulated = 0
    for individual in sorted_population[::-1]:
        fitness = apply_fitness_function(individual) + offset
        probability = fitness / normalized_fitness_sum  #fitness ratio
        accumulated += probability

        if draw <= accumulated:
            return individual

##### sprt population in ascending order of their fitness levels        
def sort_population_by_fitness(population):
    return sorted(population, key=apply_fitness_function)

######## crossover by taking a point in between the two coordinates ###########
def crossover(individual_1, individual_2):
    x1 = individual_1["x"]
    y1 = individual_1["y"]

    x2 = individual_2["x"]
    y2 = individual_2["y"]

    return {"x":round((x1 + x2)/2,3) , "y":round((y1 + y2) / 2,3)}

########### mutate witihin these bounds
def mutate(individual, x_domain, y_domain):
    next_x = individual["x"] + round(random.uniform(-0.15, 0.15),3)
    next_y = individual["y"] + round(random.uniform(-0.15, 0.15),3)
    # Guarantee we keep inside boundaries
    mutated_x = min(max(next_x, x_domain[0]), x_domain[1])
    mutated_y = min(max(next_y, y_domain[0]), y_domain[1])

    return {"x": mutated_x, "y": mutated_y}


def make_next_generation(previous_population, x_domain, y_domain):
    mutation_rate, elite_size=0.2,2
    next_generation = []
    sorted_by_fitness_population = sort_population_by_fitness(previous_population)
    population_size = len(previous_population)
    fitness_sum = sum(apply_fitness_function(individual) for individual in previous_population)

    for i in range(population_size-elite_size):
        first_choice = choice_by_roulette(sorted_by_fitness_population, fitness_sum)
        second_choice = choice_by_roulette(sorted_by_fitness_population, fitness_sum)
        individual = crossover(first_choice, second_choice)
        if random.random()>mutation_rate:
            individual = mutate(individual,x_domain,y_domain)
        next_generation.append(individual)
    next_generation.extend(sorted_by_fitness_population[population_size-elite_size:])
    return next_generation

generations = 100
x_domain=(-4, 4)
y_domain=(-4, 4)
population_size=10

def evolve(generations,x_domain, y_domain):
    i = 1
    progress=[]
    population = generate_population(population_size, x_domain, y_domain)
    while True:
        print(f" GENERATION {i}")
        flag=False
        for individual in population: 
            z=apply_fitness_function(individual)
            print(individual,z )
            #if z==5.0:
            #    flag=True
        progress.append(apply_fitness_function(sort_population_by_fitness(population)[-1]))
        if i == generations: #or flag==True:
            return population, progress
        i += 1
        population = make_next_generation(population, x_domain,y_domain)

population, progress= evolve(generations, x_domain, y_domain)
best_individual = sort_population_by_fitness(population)[-1]
print("\n FINAL RESULT")
print(best_individual, apply_fitness_function(best_individual))
plt.plot(progress)
plt.ylabel('Fitness Score')
plt.xlabel('Generation')
plt.show()