import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_fn

from diffusers import StableDiffusionPipeline
from DeepCache import DeepCacheSDHelper
from search_ga import generate_image, create_individual, crossover, mutate

def compute_fitness(img, baseline_image, interval_seq, num_steps, weight_ssim=100.0, weight_psnr=1.0, weight_eval=10.0):
    mse = F.mse_loss(img, baseline_image).item()
    img_np = img.cpu().float().numpy()
    base_np = baseline_image.cpu().float().numpy()
    
    if mse == 0:
        psnr = 100.0
    else:
        psnr = -10.0 * np.log10(mse)
        
    ssim_score = ssim_fn(base_np, img_np, data_range=1.0, channel_axis=0)
    
    eval_ratio = len(interval_seq) / float(num_steps)
    return (ssim_score * weight_ssim) + (psnr * weight_psnr) - (eval_ratio * weight_eval)

def random_search(pipe, helper, baseline, prompt, seed, steps, max_evals=15):
    best_fit = -float('inf')
    for _ in range(max_evals):
        ind = create_individual(steps)
        img = generate_image(pipe, helper, prompt, seed, steps, interval_seq=ind)
        fit = compute_fitness(img, baseline, ind, steps)
        if fit > best_fit:
            best_fit = fit
    return best_fit

def standard_ga(pipe, helper, baseline, prompt, seed, steps, pop_size=5, generations=3):
    population = [create_individual(steps) for _ in range(pop_size)]
    best_fit = -float('inf')
    
    for _ in range(generations):
        fitnesses = []
        for ind in population:
            img = generate_image(pipe, helper, prompt, seed, steps, interval_seq=ind)
            fit = compute_fitness(img, baseline, ind, steps)
            fitnesses.append(fit)
            if fit > best_fit:
                best_fit = fit
                
        new_pop = []
        new_pop.append(population[np.argmax(fitnesses)]) # Elite
        while len(new_pop) < pop_size:
            i1, i2 = random.randint(0, pop_size-1), random.randint(0, pop_size-1)
            p1 = population[i1] if fitnesses[i1] > fitnesses[i2] else population[i2]
            i1, i2 = random.randint(0, pop_size-1), random.randint(0, pop_size-1)
            p2 = population[i1] if fitnesses[i1] > fitnesses[i2] else population[i2]
            
            child = mutate(crossover(p1, p2, steps), steps, 0.1)
            new_pop.append(child)
        population = new_pop
        
    return best_fit

def island_ga(pipe, helper, baseline, prompt, seed, steps, num_islands=2, island_pop=3, generations=3):
    islands = [[create_individual(steps) for _ in range(island_pop)] for _ in range(num_islands)]
    best_fit = -float('inf')
    
    for gen in range(generations):
        all_fitness = []
        for i in range(num_islands):
            pop = islands[i]
            fitnesses = []
            for ind in pop:
                img = generate_image(pipe, helper, prompt, seed, steps, interval_seq=ind)
                fit = compute_fitness(img, baseline, ind, steps)
                fitnesses.append(fit)
                if fit > best_fit:
                    best_fit = fit
            all_fitness.append(fitnesses)
            
            new_pop = []
            new_pop.append(pop[np.argmax(fitnesses)]) # Elite
            while len(new_pop) < island_pop:
                i1, i2 = random.randint(0, island_pop-1), random.randint(0, island_pop-1)
                p1 = pop[i1] if fitnesses[i1] > fitnesses[i2] else pop[i2]
                i1, i2 = random.randint(0, island_pop-1), random.randint(0, island_pop-1)
                p2 = pop[i1] if fitnesses[i1] > fitnesses[i2] else pop[i2]
                child = mutate(crossover(p1, p2, steps), steps, 0.1)
                new_pop.append(child)
            islands[i] = new_pop
            
        # Migrate 1
        if num_islands > 1:
            m1 = islands[0][np.argmax(all_fitness[0])]
            m2 = islands[1][np.argmax(all_fitness[1])]
            islands[1][np.argmin(all_fitness[1])] = m1
            islands[0][np.argmin(all_fitness[0])] = m2

    return best_fit

def main():
    print("Loading SD pipeline to run DeepCache benchmarks (Fast mode)...")
    pipe = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16
    ).to("cuda:0")
    pipe.set_progress_bar_config(disable=True)
    helper = DeepCacheSDHelper(pipe=pipe)
    
    prompt = "a photo of an astronaut on a moon"
    seed = 42
    steps = 20 # Keeping steps low for fast evaluation
    
    print("Generating baseline...")
    baseline = generate_image(pipe, helper, prompt, seed, steps, interval_seq=None)
    
    print("Running Random Search (18 evals)...")
    rand_fit = random_search(pipe, helper, baseline, prompt, seed, steps, max_evals=18)
    
    print("Running Standard GA (Pop 6, 3 Gens = 18 evals)...")
    std_fit = standard_ga(pipe, helper, baseline, prompt, seed, steps, pop_size=6, generations=3)
    
    print("Running Island GA (2 Islands, Pop 3, 3 Gens = 18 evals)...")
    isl_fit = island_ga(pipe, helper, baseline, prompt, seed, steps, num_islands=2, island_pop=3, generations=3)
    
    print("\n[RESULT_TABLE]")
    print(f"{'Task':<20} | {'Random Search':<15} | {'Standard GA':<15} | {'Island GA':<15}")
    print("-" * 75)
    print(f"{'DeepCache Schedule (Fitness)':<20} | {rand_fit:<15.4f} | {std_fit:<15.4f} | {isl_fit:<15.4f}")
    print("[END_TABLE]")

if __name__ == '__main__':
    main()
