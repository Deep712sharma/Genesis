import random
import numpy as np
import sys
import copy
from deap import base, creator, tools, benchmarks

NDIM = 10
POP_SIZE = 100
NGEN = 50

# Custom Beale function (2D)
def beale(individual):
    x, y = individual[0], individual[1]
    term1 = (1.5 - x + x*y)**2
    term2 = (2.25 - x + x*y**2)**2
    term3 = (2.625 - x + x*y**3)**2
    return (term1 + term2 + term3,)

def get_benchmark(name):
    if name == 'Rosenbrock': return lambda ind: benchmarks.rosenbrock(ind)
    elif name == 'Beale': return beale
    elif name == 'Himmelblau': 
        return lambda ind: benchmarks.himmelblau([ind[0], ind[1]])
    elif name == 'Ackley': return lambda ind: benchmarks.ackley(ind)
    elif name == 'Rastrigin': return lambda ind: benchmarks.rastrigin(ind)

def get_ndim(name):
    if name in ['Beale', 'Himmelblau']: return 2
    return NDIM

# We initialize creator classes outside to avoid re-creation warnings
try:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("ContinuousIndividual", list, fitness=creator.FitnessMin, cached_fitness=None)
    creator.create("MetaFitnessMax", base.Fitness, weights=(1.0,))
    creator.create("MetaIndividual", list, fitness=creator.MetaFitnessMax)
except Exception:
    pass

def run_inner_ga(func, ndim, interval_seq, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -5, 5)
    toolbox.register("individual", tools.initRepeat, creator.ContinuousIndividual, toolbox.attr_float, n=ndim)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", func)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=-5, up=5, eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=-5, up=5, eta=20.0, indpb=1.0/ndim)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    pop = toolbox.population(n=POP_SIZE)
    
    # Gen 0 Evaluation
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        ind.cached_fitness = fit
    
    for gen in range(1, NGEN):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.8:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
                
        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values
                
        # Deepcache Logic
        if gen in interval_seq:
            # Exact Evaluation step
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                ind.cached_fitness = fit
        else:
            # Surrogate Cached step: skip actual function, use parents cached fitness
            for ind in offspring:
                if not ind.fitness.valid:
                    if getattr(ind, 'cached_fitness', None) is not None:
                        ind.fitness.values = ind.cached_fitness
                    else:
                        fit = toolbox.evaluate(ind)
                        ind.fitness.values = fit
                        ind.cached_fitness = fit
        
        pop[:] = offspring
        
    return pop

def calculate_entropy(population):
    # Sort and take top-64 elites
    population = sorted(population, key=lambda x: x.fitness.values[0])[:64]
    data = np.array([list(ind) for ind in population])
    
    bins = np.linspace(-5, 5, 20)
    total_entropy = 0.0
    for dim in range(data.shape[1]):
        hist, _ = np.histogram(data[:, dim], bins=bins)
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]
        total_entropy -= np.sum(prob * np.log(prob))
        
    # Scale up mathematically to fit around DEAP typical string values reported
    return float(total_entropy)

def calculate_fitness(population):
    population = sorted(population, key=lambda x: x.fitness.values[0])[:64]
    mean_val = np.mean([ind.fitness.values[0] for ind in population])
    # Standard normalization: Best is 1.0 (min=0)
    return float(1.0 / (1.0 + mean_val))

def run_meta_ga(func, ndim):
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.MetaIndividual, toolbox.attr_bool, n=NGEN)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def eval_meta(individual):
        seq = [i for i, b in enumerate(individual) if b == 1]
        if 0 not in seq: seq.insert(0, 0)
        pop = run_inner_ga(func, ndim, seq, seed=42)
        fit_score = calculate_fitness(pop)
        
        # Penalize over-evaluating exact function
        eval_ratio = len(seq) / float(NGEN)
        return (fit_score - 0.5 * eval_ratio,)

    toolbox.register("evaluate", eval_meta)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # very fast meta setup
    pop = toolbox.population(n=8) 
    
    for gen in range(3):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
                
        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values
                
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        pop[:] = offspring

    best_ind = tools.selBest(pop, 1)[0]
    return [i for i, b in enumerate(best_ind) if b == 1]

def main():
    tasks = ['Rosenbrock', 'Beale', 'Himmelblau', 'Ackley', 'Rastrigin']
    results = {'Baseline': {}, 'DeepCache': {}, 'Ga_Search': {}}
    
    seq_baseline = list(range(NGEN))
    seq_deepcache = list(range(0, NGEN, 3))
    
    print("Beginning Continual DeepCache Evaluation on Benchmarks...")
    for task in tasks:
        print(f" -> Mapping to {task} surrogate functions...", flush=True)
        func = get_benchmark(task)
        ndim = get_ndim(task)
        
        # Baseline
        pop_base = run_inner_ga(func, ndim, seq_baseline, seed=123)
        results['Baseline'][task] = (calculate_entropy(pop_base), calculate_fitness(pop_base))
        
        # DeepCache
        pop_dc = run_inner_ga(func, ndim, seq_deepcache, seed=123)
        results['DeepCache'][task] = (calculate_entropy(pop_dc), calculate_fitness(pop_dc))
        
        # Ga_Search
        seq_ga = run_meta_ga(func, ndim)
        if 0 not in seq_ga: seq_ga.insert(0, 0)
        pop_ga = run_inner_ga(func, ndim, seq_ga, seed=123)
        results['Ga_Search'][task] = (calculate_entropy(pop_ga), calculate_fitness(pop_ga))
        
    print("\n[RESULT_TABLE]")
    print("| Task | Baseline | DeepCache | Ga_Search |")
    print("| :--- | :---: | :---: | :---: |")
    for task in tasks:
        e_b, f_b = results['Baseline'][task]
        e_d, f_d = results['DeepCache'][task]
        e_g, f_g = results['Ga_Search'][task]
        
        # format similar to image
        b_str = f"{e_b:.2f} ({f_b:.2f})"
        d_str = f"{e_d:.2f} ({f_d:.2f})"
        g_str = f"{e_g:.2f} ({f_g:.2f})"
        print(f"| {task} | {b_str} | {d_str} | {g_str} |")
    print("[END_TABLE]")

if __name__ == '__main__':
    main()
