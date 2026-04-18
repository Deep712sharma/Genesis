import argparse
import time
import random
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim_fn

from DeepCache import DeepCacheSDHelper
from diffusers import StableDiffusionPipeline
from search_ga import create_individual, crossover, mutate


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Generate single image using GA, DeepCache, and Original pipeline for comparison.")
    parser.add_argument("--prompt", type=str, default="a photo of an astronaut riding a horse on mars", help="Text prompt for image generation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    # DeepCache (Standard) params
    parser.add_argument("--cache_interval", type=int, default=3, help="Cache interval for standard DeepCache.")
    parser.add_argument("--cache_branch_id", type=int, default=0, help="Cache branch id for DeepCache.")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps.")
    
    # GA Params
    parser.add_argument("--pop_size", type=int, default=5, help="Population size for GA search per island.")
    parser.add_argument("--generations", type=int, default=3, help="Number of generations for GA search.")
    parser.add_argument("--num_islands", type=int, default=3, help="Number of independent islands.")
    parser.add_argument("--migration_interval", type=int, default=2, help="Generations between migrations.")
    parser.add_argument("--migration_size", type=int, default=1, help="Number of individuals to migrate.")
    
    args = parser.parse_args()

    print("Loading Stable Diffusion v1.5 Model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16
    ).to("cuda:0")

    prompt = args.prompt
    seed = args.seed

    print("Warming up GPU...")
    for _ in range(1):
        set_random_seed(seed)
        _ = pipe(prompt, num_inference_steps=5, output_type='pt')

    # 1. Original Pipeline (Baseline for GA)
    print(f"\n{'='*40}\nRunning Original Pipeline (Baseline)...")
    set_random_seed(seed)
    start_time = time.time()
    pipeline_output_original = pipe(
        prompt, 
        num_inference_steps=args.steps,
        output_type='pt'
    ).images[0]
    origin_time = time.time() - start_time
    save_image([pipeline_output_original], 'text2img_original.png')

    helper = DeepCacheSDHelper(pipe=pipe)

    # Run GA Search to find interval sequence dynamically
    print(f"\n{'='*40}\nRunning GA Search to find optimal interval sequence (Island Model)...")
    print(f"Islands: {args.num_islands}, Population per island: {args.pop_size}, Generations: {args.generations}")
    ga_start_time = time.time()
    
    islands = [[create_individual(args.steps) for _ in range(args.pop_size)] for _ in range(args.num_islands)]
    best_ind = None
    best_fitness = -float('inf')
    
    for gen in range(args.generations):
        all_island_fitnesses = []
        
        for island_idx in range(args.num_islands):
            population = islands[island_idx]
            fitnesses = []
            
            for i, ind in enumerate(population):
                helper.set_params(interval_seq=ind, cache_branch_id=args.cache_branch_id)
                helper.enable()
                set_random_seed(seed)
                img = pipe(prompt, num_inference_steps=args.steps, output_type='pt').images[0]
                helper.disable()
                
                # calculate fitness against Original
                mse = F.mse_loss(img, pipeline_output_original).item()
                psnr = 100.0 if mse == 0 else -10.0 * np.log10(mse)
                
                img_np = img.cpu().float().numpy()
                base_np = pipeline_output_original.cpu().float().numpy()
                ssim_score = ssim_fn(base_np, img_np, data_range=1.0, channel_axis=0)
                
                eval_ratio = len(ind) / float(args.steps)
                fit = (ssim_score * 1.0) + (psnr * 1.0) - (eval_ratio * 1.0)
                fitnesses.append(fit)
                
                if fit > best_fitness:
                    best_fitness = fit
                    best_ind = ind
                    
            all_island_fitnesses.append(fitnesses)
            
            # Next generation for this island
            sorted_indices = np.argsort(fitnesses)[::-1]
            new_pop = [population[sorted_indices[0]]] # elitism (top 1)
            
            while len(new_pop) < args.pop_size:
                t1, t2 = random.randint(0, args.pop_size-1), random.randint(0, args.pop_size-1)
                p1 = population[t1] if fitnesses[t1] > fitnesses[t2] else population[t2]
                
                t1, t2 = random.randint(0, args.pop_size-1), random.randint(0, args.pop_size-1)
                p2 = population[t1] if fitnesses[t1] > fitnesses[t2] else population[t2]
                
                child = crossover(p1, p2, args.steps)
                child = mutate(child, args.steps, mutation_rate=0.1)
                new_pop.append(child)
                
            islands[island_idx] = new_pop
            
        print(f"Gen {gen+1}/{args.generations} - Best Overall Fitness: {best_fitness:.2f}")

        # Migration logic
        if gen > 0 and (gen + 1) % args.migration_interval == 0 and args.num_islands > 1:
            migrants = []
            for island_idx in range(args.num_islands):
                fitness_scores = all_island_fitnesses[island_idx]
                sorted_indices = np.argsort(fitness_scores)[::-1]
                island_migrants = [islands[island_idx][idx] for idx in sorted_indices[:args.migration_size]]
                migrants.append(island_migrants)
                
            for island_idx in range(args.num_islands):
                next_island = (island_idx + 1) % args.num_islands
                next_fitness_scores = all_island_fitnesses[next_island]
                
                worst_indices = np.argsort(next_fitness_scores)
                for i in range(args.migration_size):
                    replace_idx = worst_indices[i]
                    islands[next_island][replace_idx] = migrants[island_idx][i]
            
    ga_search_time = time.time() - ga_start_time
    interval_seq = best_ind
    print(f"\nGA Search Completed in {ga_search_time:.2f} seconds.")
    print(f"Found Best Interval Sequence: {interval_seq}")

    # 2. Standard DeepCache
    print(f"\n{'='*40}\nEnable Standard DeepCache with static interval {args.cache_interval}...")
    helper.set_params(
        cache_interval=args.cache_interval,
        cache_branch_id=args.cache_branch_id,
        interval_seq=None # Ensure interval_seq is implicitly None
    )
    helper.enable()

    print("Running Pipeline with Standard DeepCache...")
    set_random_seed(seed)
    start_time = time.time()
    deepcache_pipeline_output = pipe(
        prompt,
        num_inference_steps=args.steps,
        output_type='pt'
    ).images[0]
    deepcache_time = time.time() - start_time
    save_image([deepcache_pipeline_output], 'text2img_deepcache_standard.png')
    helper.disable()

    # 3. GA DeepCache
    print(f"\n{'='*40}\nEnable GA DeepCache with searched interval sequence...")
    helper.set_params(
        interval_seq=interval_seq,
        cache_branch_id=args.cache_branch_id,
    )
    helper.enable()

    print("Running Pipeline with GA-optimized DeepCache...")
    set_random_seed(seed)
    start_time = time.time()
    ga_pipeline_output = pipe(
        prompt,
        num_inference_steps=args.steps,
        output_type='pt'
    ).images[0]
    ga_time = time.time() - start_time
    save_image([ga_pipeline_output], 'text2img_deepcache_ga.png')
    helper.disable()

    print(f"\n{'='*40}")
    print("Execution Time Comparison:")
    print(f"Prompt: '{prompt}'")
    print(f"Steps:  {args.steps}")
    print(f"GA Search Duration: {ga_search_time:.2f} seconds")
    print("-" * 50)
    print(f"Original Pipeline:               {origin_time:.2f} seconds")
    print(f"Standard DeepCache (interval {args.cache_interval}):  {deepcache_time:.2f} seconds (Speedup: {origin_time/deepcache_time:.2f}x)")
    print(f"GA DeepCache (sequence):         {ga_time:.2f} seconds (Speedup: {origin_time/ga_time:.2f}x)")
    print("-" * 50)
    print("Generated images have been saved to:")
    print("- text2img_original.png")
    print("- text2img_deepcache_standard.png")
    print("- text2img_deepcache_ga.png")


if __name__ == "__main__":
    main()
