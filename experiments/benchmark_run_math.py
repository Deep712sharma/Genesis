import numpy as np
import time
import random

# Math benchmark functions (dimension = 10)
NDIM = 10

def sphere(x):
    return np.sum(x**2)

def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(x*x) / len(x)))
    cos_term = -np.exp(np.sum(np.cos(c * x) / len(x)))
    return a + np.exp(1) + sum_sq_term + cos_term

def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

functions = {
    'Sphere': sphere,
    'Ackley': ackley,
    'Rastrigin': rastrigin
}

POP_SIZE = 50
GENERATIONS = 50

# --- Search Algorithms ---

def random_search(func, bounds=(-5.12, 5.12)):
    best_fit = float('inf')
    evals = 0
    start = time.time()
    for _ in range(POP_SIZE * GENERATIONS):
        x = np.random.uniform(bounds[0], bounds[1], NDIM)
        fit = func(x)
        if fit < best_fit:
            best_fit = fit
        evals += 1
    return best_fit, time.time() - start

def standard_ga(func, bounds=(-5.12, 5.12)):
    # Standard GA
    pop = np.random.uniform(bounds[0], bounds[1], (POP_SIZE, NDIM))
    best_fit = float('inf')
    
    start = time.time()
    for gen in range(GENERATIONS):
        fitnesses = np.array([func(ind) for ind in pop])
        
        best_idx = np.argmin(fitnesses)
        if fitnesses[best_idx] < best_fit:
            best_fit = fitnesses[best_idx]
            
        # Tournament selection
        new_pop = []
        for _ in range(POP_SIZE):
            idx1, idx2 = random.randint(0, POP_SIZE-1), random.randint(0, POP_SIZE-1)
            parent1 = pop[idx1] if fitnesses[idx1] < fitnesses[idx2] else pop[idx2]
            
            idx1, idx2 = random.randint(0, POP_SIZE-1), random.randint(0, POP_SIZE-1)
            parent2 = pop[idx1] if fitnesses[idx1] < fitnesses[idx2] else pop[idx2]
            
            # Crossover
            mask = np.random.rand(NDIM) < 0.5
            child = np.where(mask, parent1, parent2)
            
            # Mutate
            mutate_mask = np.random.rand(NDIM) < 0.1
            noise = np.random.normal(0, 0.5, NDIM)
            child = np.where(mutate_mask, child + noise, child)
            child = np.clip(child, bounds[0], bounds[1])
            new_pop.append(child)
            
        pop = np.array(new_pop)
    return best_fit, time.time() - start

def island_ga(func, bounds=(-5.12, 5.12)):
    # Island Model GA (Like the one in project)
    NUM_ISLANDS = 3
    ISLAND_POP = max(2, POP_SIZE // NUM_ISLANDS) # ~16 per island
    MIGRATION_INTERVAL = 5
    MIGRATION_SIZE = 2
    
    islands = [np.random.uniform(bounds[0], bounds[1], (ISLAND_POP, NDIM)) for _ in range(NUM_ISLANDS)]
    best_fit = float('inf')
    
    start = time.time()
    for gen in range(GENERATIONS):
        fitness_per_island = []
        for i in range(NUM_ISLANDS):
            pop = islands[i]
            fitnesses = np.array([func(ind) for ind in pop])
            fitness_per_island.append(fitnesses)
            
            best_idx = np.argmin(fitnesses)
            if fitnesses[best_idx] < best_fit:
                best_fit = fitnesses[best_idx]
            
            # Selection & Crossover
            new_pop = []
            for _ in range(ISLAND_POP):
                i1, i2 = random.randint(0, ISLAND_POP-1), random.randint(0, ISLAND_POP-1)
                p1 = pop[i1] if fitnesses[i1] < fitnesses[i2] else pop[i2]
                
                i1, i2 = random.randint(0, ISLAND_POP-1), random.randint(0, ISLAND_POP-1)
                p2 = pop[i1] if fitnesses[i1] < fitnesses[i2] else pop[i2]
                
                mask = np.random.rand(NDIM) < 0.5
                child = np.where(mask, p1, p2)
                
                mutate_mask = np.random.rand(NDIM) < 0.1
                noise = np.random.normal(0, 0.5, NDIM)
                child = np.where(mutate_mask, child + noise, child)
                child = np.clip(child, bounds[0], bounds[1])
                new_pop.append(child)
            islands[i] = np.array(new_pop)
            
        # Migration
        if gen > 0 and gen % MIGRATION_INTERVAL == 0:
            migrants = []
            for i in range(NUM_ISLANDS):
                best_indices = np.argsort(fitness_per_island[i])[:MIGRATION_SIZE]
                migrants.append(islands[i][best_indices])
                
            for i in range(NUM_ISLANDS):
                next_i = (i + 1) % NUM_ISLANDS
                worst_indices = np.argsort(fitness_per_island[next_i])[-MIGRATION_SIZE:]
                for m_idx in range(MIGRATION_SIZE):
                    islands[next_i][worst_indices[m_idx]] = migrants[i][m_idx]
                    
    return best_fit, time.time() - start

def main():
    print("Running Mathematical Benchmarks...\n")
    print(f"{'Function':<12} | {'Random Search Best':<20} | {'Standard GA Best':<20} | {'Island GA Best':<20}")
    print("-" * 80)
    for name, func in functions.items():
        random_fit, r_time = random_search(func)
        std_ga_fit, s_time = standard_ga(func)
        isl_ga_fit, i_time = island_ga(func)
        
        print(f"{name:<12} | {random_fit:<20.4f} | {std_ga_fit:<20.4f} | {isl_ga_fit:<20.4f}")
        
if __name__ == '__main__':
    main()
