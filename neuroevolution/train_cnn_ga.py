'''Trains a convolutional neural network on sample images from the environment
using neuroevolution to maximize the ability to discriminate between input
images.

Reference:
Koutnik, Jan, Jurgen Schmidhuber, and Faustino Gomez. "Evolving deep
unsupervised convolutional networks for vision-based reinforcement
learning." Proceedings of the 2014 conference on Genetic and
evolutionary computation. ACM, 2014.
'''

import random
import numpy as np
from cnn import create_cnn, calculate_cnn_output, calculate_fitness
from nn_utilities import update_model_weights
from datasets import load_images_torcs_4
from visualization import plot_feature_vectors
from deap import algorithms, base, creator, tools
from operator import attrgetter
from matplotlib import pyplot as plt

# Set the following parameters:
OUTPUT_DIR = 'experiments/train_cnn_ga_11/'
images = load_images_torcs_4()

# Create the ConvNet and load the training set
model = create_cnn()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

INDIVIDUAL_SIZE = 993

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -1.5, 1.5)
toolbox.register("individual",
                 tools.initRepeat,
                 creator.Individual,
                 toolbox.attr_float,
                 n=INDIVIDUAL_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def ga_fitness(individual):
    # Update the ConvNet parameters
    update_model_weights(model, np.asarray(individual))

    # Calculate the output feature vectors
    feature_vectors = calculate_cnn_output(model, images)

    # Check their fitness
    fitness = calculate_fitness(feature_vectors)

    return fitness,

toolbox.register("evaluate", ga_fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.5, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=10)  # Optimize this hyperparameter


# This is a modified version of the eaSimple algorithm included with DEAP here:
# https://github.com/DEAP/deap/blob/master/deap/algorithms.py#L84
def eaSimpleModified(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    best = []

    best_ind = max(population, key=attrgetter("fitness"))
    best.append(best_ind)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Save the best individual from the generation
        best_ind = max(offspring, key=attrgetter("fitness"))
        best.append(best_ind)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook, best

def run(num_gen=10,
        n=100,
        mutpb=0.8,
        cxpb=0.5):
    np.random.seed(0)
    history = tools.History()
    # Decorate the variation operators
    toolbox.decorate("mate", history.decorator)
    toolbox.decorate("mutate", history.decorator)

    pop = toolbox.population(n=n)
    history.update(pop)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log, best = eaSimpleModified(pop,
                                   toolbox,
                                   cxpb=cxpb,
                                   mutpb=mutpb,
                                   ngen=num_gen,
                                   stats=stats,
                                   halloffame=hof,
                                   verbose=True)

    return pop, log, hof, history, best


def plot_results(filename,
                 gen,
                 fitness_maxs,
                 fitness_avgs):
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fitness_maxs, "r-", label="Maximum Fitness")
    line2 = ax1.plot(gen, fitness_avgs, "b-", label="Average Fitness")
    lines = line1 + line2
    labs = [line.get_label() for line in lines]
    ax1.legend(lines, labs, loc="lower right")
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    plt.savefig('{}'.format(filename))


def run_experiments(output_dir):
    POPULATION_SIZE = 100
    NUM_GENERATIONS = 100
    CROSSOVER_PROB = 0.5

    MUTATION_PROBS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

    for mutation_prob in MUTATION_PROBS:
        pop, log, hof, history, best_per_gen = run(num_gen=NUM_GENERATIONS,
                                               n=POPULATION_SIZE,
                                               cxpb=CROSSOVER_PROB,
                                               mutpb=mutation_prob)
        best = np.asarray(hof)
        gen = log.select("gen")
        fitness_maxs = log.select("max")
        fitness_avgs = log.select("avg")
        plot_results(filename='{}train_cnn_ga_mutpb_{}.png'.
                     format(output_dir,
                            str(mutation_prob).replace('.', '_')),
                     gen=gen,
                     fitness_maxs=fitness_maxs,
                     fitness_avgs=fitness_avgs)

        np.savetxt('{}train_cnn_ga_mutpb_{}.out'.
                   format(output_dir,
                          str(mutation_prob).replace('.', '_')), best)

        # Plot the feature vectors produced by the best individual from each
        # generation
        for gen in range(len(best_per_gen)):
            update_model_weights(model, np.asarray(best_per_gen[gen]))
            feature_vectors = calculate_cnn_output(model, images)
            plot_feature_vectors(feature_vectors, filename='{}feature_vectors_{}__{}.png'.\
                format(output_dir, str(mutation_prob).replace('.', '_'), gen))


if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    run_experiments(output_dir=OUTPUT_DIR)
