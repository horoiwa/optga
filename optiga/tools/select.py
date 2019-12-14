from optiga.tools.nsga2 import NSGA2


def SelectNSGA2(population, fitness, size):
    next_population = NSGA2(population, fitness, size)
    return next_population
