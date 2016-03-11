'''Neuroevolution procedure: evolves a population of TORCS RNN controllers.
'''

import numpy as np
import random
from deap import algorithms, base, creator, tools
import multiprocessing
from policy import CNNRNNPolicy
from nn_utilities import update_model_weights
from deap import creator
import scoop
from scoop import futures
import code, traceback, signal

# Import the client definition for the simulation environment
from torcs_controller import pyclient

# Define the following parameter:
PRETRAINED_WEIGHTS_CNN = 'experiments/train_cnn_ga_11/train_cnn_ga_mutpb_0_2.out'


def ga_fitness(individual):
    sessionid  = multiprocessing.current_process()._identity[0]

    # Instantiate the control policy
    # Load the ConvNet weights
    pretrained_weights_cnn = \
         np.loadtxt('{}'.format(PRETRAINED_WEIGHTS_CNN))

    POLICY = CNNRNNPolicy(cnn_weights=pretrained_weights_cnn)

    # Instantiate the simulator environment
    TORCS = pyclient.TORCS(episode_length=500,
                           policy=POLICY,
                           sessionid=sessionid)

    # The GA will update the RNN parameters to evaluate candidate solutions
    update_model_weights(POLICY.rnn, np.asarray(individual))
    POLICY.rnn.reset_states()

    # Run an episode on the track and its mirror image and measure the average
    # fitness
    fitness = 0

    fitness1 = TORCS.run(track_id=1)
    fitness2 = TORCS.run(track_id=2)

    fitness = min(fitness1, fitness2)

    return fitness,


def run(num_gen,
        n,
        mutpb,
        cxpb):
    """
    Runs multiple episodes, evolving the RNN parameters using a GA
    """
    history = tools.History()
    # Decorate the variation operators
    toolbox.decorate("mate", history.decorator)
    toolbox.decorate("mutate", history.decorator)

    pool = multiprocessing.Pool(processes=12)
    toolbox.register("map", pool.map)

    pop = toolbox.population(n=n)
    history.update(pop)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop,
                                   toolbox,
                                   cxpb=cxpb,
                                   mutpb=mutpb,
                                   ngen=num_gen,
                                   stats=stats,
                                   halloffame=hof,
                                   verbose=True)

    return pop, log, hof, history


# Set up the genetic algorithm to evolve the RNN parameters
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

INDIVIDUAL_SIZE = 33

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -1.5, 1.5)
toolbox.register("individual",
                 tools.initRepeat,
                 creator.Individual,
                 toolbox.attr_float,
                 n=INDIVIDUAL_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", ga_fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.5, indpb=0.10)

# Use tournament selection: choose a subset consisting of k members of that
# population, and from that subset, choose the best individual
toolbox.register("select", tools.selTournament, tournsize=2)


def main():
    try:
        NUM_GENERATIONS = 100
        POPULATION_SIZE = 96
        MUTATION_PROB = 0.02
        CROSSOVER_PROB = 0.5

        pop, log, hof, history = \
            run(num_gen=NUM_GENERATIONS,
                n=POPULATION_SIZE,
                cxpb=CROSSOVER_PROB,
                mutpb=MUTATION_PROB)
                
        best = np.asarray(hof)
        gen = log.select("gen")
        fitness_maxs = log.select("max")
        fitness_avgs = log.select("avg")

        # Plot the results
        from matplotlib import pyplot as plt
        plt.plot(fitness_maxs)#, '.')
        plt.plot(fitness_avgs)#, '.')
        plt.legend(['maximum', 'average'], loc=4)
        plt.xlabel('Episode')
        plt.ylabel('Fitness')

        # Save the results to disk
        np.savetxt('weights.out', best)
        np.savetxt('fitness_avgs.out', fitness_avgs)
        np.savetxt('fitness_maxs.out', fitness_maxs)

        individuals = []
        for i in history.genealogy_history.items():
            individuals.append(i[1])
        inp = np.array(individuals)
        np.savetxt('history.out', inp)

        plt.savefig('learning_history.png')
        plt.show()

        import IPython
        IPython.embed()
    finally:
        pass

if __name__ == '__main__':
    main()
