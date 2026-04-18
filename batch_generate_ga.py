import os
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
    prompts = [
        "a doorknocker shaped like a lion's head",
        "A portrait of an old man",
        "the silhouette of an elephant on the full moon",
        "A beaver wearing glasses, stands next to a stack of books.",
        "A oil painting of a badger sniffing a yellow rose.",
        "A raccoon wearing formal clothes, wearing a tophat.",
        "A black and white landscape photograph of a black tree",
        "a Styracosaurus displaying its horns",
        "A figure shrouded in mists peers up cobble stone street, there should be a person like figure on the road in the mist."
    ]
    
    seed = 42
    steps = 50
    pop_size = 5
    generations = 3
    num_islands = 3
    migration_interval = 2
    migration_size = 1

    print("Loading Stable Diffusion v1.5 Model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16
    ).to("cuda:0")

    helper = DeepCacheSDHelper(pipe=pipe)
    
    out_dir = "ga_generated_images"
    os.makedirs(out_dir, exist_ok=True)

    for p_idx, prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"Processing Prompt {p_idx+1}/{len(prompts)}: {prompt}")
        
        # 1. Base Image
        set_random_seed(seed)
        pipeline_output_original = pipe(
            prompt, 
            num_inference_steps=steps,
            output_type='pt'
        ).images[0]
        # save_image([pipeline_output_original], f'{out_dir}/img_{p_idx+1}_baseline.png')

        # 2. run GA 
        islands = [[create_individual(steps) for _ in range(pop_size)] for _ in range(num_islands)]
        best_ind = None
        best_fitness = -float('inf')
        
        for gen in range(generations):
            all_island_fitnesses = []
            for island_idx in range(num_islands):
                population = islands[island_idx]
                fitnesses = []
                for i, ind in enumerate(population):
                    helper.set_params(interval_seq=ind, cache_branch_id=0)
                    helper.enable()
                    set_random_seed(seed)
                    img = pipe(prompt, num_inference_steps=steps, output_type='pt').images[0]
                    helper.disable()
                    
                    mse = F.mse_loss(img, pipeline_output_original).item()
                    psnr = 100.0 if mse == 0 else -10.0 * np.log10(mse)
                    img_np = img.cpu().float().numpy()
                    base_np = pipeline_output_original.cpu().float().numpy()
                    ssim_score = ssim_fn(base_np, img_np, data_range=1.0, channel_axis=0)
                    
                    eval_ratio = len(ind) / float(steps)
                    fit = (ssim_score * 1.0) + (psnr * 1.0) - (eval_ratio * 1.0)
                    fitnesses.append(fit)
                    
                    if fit > best_fitness:
                        best_fitness = fit
                        best_ind = ind
                        
                all_island_fitnesses.append(fitnesses)
                
                # Next generation
                sorted_indices = np.argsort(fitnesses)[::-1]
                new_pop = [population[sorted_indices[0]]]
                while len(new_pop) < pop_size:
                    t1, t2 = random.randint(0, pop_size-1), random.randint(0, pop_size-1)
                    p1 = population[t1] if fitnesses[t1] > fitnesses[t2] else population[t2]
                    t1, t2 = random.randint(0, pop_size-1), random.randint(0, pop_size-1)
                    p2 = population[t1] if fitnesses[t1] > fitnesses[t2] else population[t2]
                    child = crossover(p1, p2, steps)
                    child = mutate(child, steps, mutation_rate=0.1)
                    new_pop.append(child)
                islands[island_idx] = new_pop
                
            # Migration
            if gen > 0 and (gen + 1) % migration_interval == 0 and num_islands > 1:
                migrants = []
                for island_idx in range(num_islands):
                    fitness_scores = all_island_fitnesses[island_idx]
                    sorted_indices = np.argsort(fitness_scores)[::-1]
                    migrants.append([islands[island_idx][idx] for idx in sorted_indices[:migration_size]])
                for island_idx in range(num_islands):
                    next_island = (island_idx + 1) % num_islands
                    next_fitness_scores = all_island_fitnesses[next_island]
                    worst_indices = np.argsort(next_fitness_scores)
                    for i in range(migration_size):
                        islands[next_island][worst_indices[i]] = migrants[island_idx][i]
                        
        print(f"Best Seq for Prompt {p_idx+1}: {best_ind} | Fitness: {best_fitness:.2f}")
        
        # 3. GA Image
        helper.set_params(interval_seq=best_ind, cache_branch_id=0)
        helper.enable()
        set_random_seed(seed)
        ga_img = pipe(prompt, num_inference_steps=steps, output_type='pt').images[0]
        helper.disable()
        
        out_path = f'{out_dir}/img_{p_idx+1}_ga.png'
        save_image([ga_img], out_path)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
