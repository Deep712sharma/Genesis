import argparse
import time
import random
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn

from diffusers import StableDiffusionPipeline
from DeepCache import DeepCacheSDHelper

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def generate_image(pipe, helper, prompt, seed, num_inference_steps, interval_seq=None):
    set_random_seed(seed)
    
    if interval_seq is not None:
        helper.set_params(interval_seq=interval_seq, cache_branch_id=0)
        helper.enable()
        
    image = pipe(
        prompt, 
        num_inference_steps=num_inference_steps,
        output_type='pt'
    ).images[0]
    
    if interval_seq is not None:
        helper.disable()
        
    return image

def compute_fitness(image, baseline_image, interval_seq, num_inference_steps, alpha=100.0):
    # Mean Squared Error
    mse = F.mse_loss(image, baseline_image).item()
    
    # We want to minimize MSE and minimize number of evaluations.
    # We maximize fitness. 
    eval_ratio = len(interval_seq) / float(num_inference_steps)
    
    # fitness = - MSE * Alpha - eval_ratio. 
    fitness = - (mse * alpha) - eval_ratio
    return fitness, mse, eval_ratio

def create_individual(num_steps):
    # Probability of evaluate = 0.5
    ind = [i for i in range(num_steps) if random.random() > 0.5 or i == 0]
    return ind

def crossover(parent1, parent2, num_steps):
    # Uniform crossover
    child = []
    for i in range(num_steps):
        p1_has = i in parent1
        p2_has = i in parent2
        
        if random.random() > 0.5:
            if p1_has: child.append(i)
        else:
            if p2_has: child.append(i)
    
    # Ensure 0 is always evaluated
    if 0 not in child:
        child.insert(0, 0)
    return sorted(list(set(child)))

def mutate(individual, num_steps, mutation_rate=0.1):
    child = []
    for i in range(num_steps):
        has_val = i in individual
        if random.random() < mutation_rate and i != 0:
            has_val = not has_val
        if has_val:
            child.append(i)
    if 0 not in child:
        child.insert(0, 0)
    return sorted(list(set(child)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default='a photo of an astronaut on a moon')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--pop_size", type=int, default=10, help="Population size per island")
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--num_islands", type=int, default=3, help="Number of independent islands")
    parser.add_argument("--migration_interval", type=int, default=2, help="Generations between migrations")
    parser.add_argument("--migration_size", type=int, default=1, help="Number of individuals to migrate")
    parser.add_argument("--weight_ssim", type=float, default=100.0, help="Weight for SSIM reward")
    parser.add_argument("--weight_psnr", type=float, default=1.0, help="Weight for PSNR reward")
    parser.add_argument("--weight_eval", type=float, default=10.0, help="Weight for eval ratio penalty")
    parser.add_argument("--dataset", type=str, default=None, choices=[None, 'coco2017', 'parti'], help="Dataset to evaluate on")
    parser.add_argument("--num_prompts", type=int, default=4, help="Number of prompts to evaluate from dataset")
    parser.add_argument("--eval_coco", action="store_true", help="Automatically run full evaluation and clip score after GA")
    parser.add_argument("--eval_prompts", type=int, default=500, help="Number of COCO prompts for post-search evaluation")
    parser.add_argument("--eval_fid", action="store_true", help="Automatically run FID evaluation using calculate_fid_coco.py after GA")
    parser.add_argument("--fid_images", type=int, default=1000, help="Number of images for FID evaluation")
    args = parser.parse_args()

    print("Loading SD pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16
    ).to("cuda:0")
    pipe.set_progress_bar_config(disable=True)
    helper = DeepCacheSDHelper(pipe=pipe)
    
    if args.dataset == 'coco2017':
        print("Loading COCO dataset...")
        from datasets import load_dataset
        dataset = load_dataset("phiyodr/coco2017")
        all_prompts = [{"Prompt": sample['captions'][0]} for sample in dataset['validation']]
        prompts = [p["Prompt"] for p in all_prompts[:args.num_prompts]]
    elif args.dataset == 'parti':
        print("Loading Parti dataset...")
        from datasets import load_dataset
        dataset = load_dataset("nateraw/parti-prompts", split="train")
        prompts = [sample['Prompt'] for sample in dataset][:args.num_prompts]
    else:
        prompts = [args.prompt]
        
    print(f"Generating {len(prompts)} baseline image(s)...")
    baseline_images = []
    for p in prompts:
        baseline_images.append(generate_image(pipe, helper, p, args.seed, args.steps, interval_seq=None))
    
    # Initialize population for each island
    islands = [[create_individual(args.steps) for _ in range(args.pop_size)] for _ in range(args.num_islands)]
    
    best_overall_individual = None
    best_overall_fitness = -float('inf')
    best_overall_ssim = None
    best_overall_psnr = None
    best_overall_evals = None
    
    for gen in range(args.generations):
        print(f"\n--- Generation {gen + 1}/{args.generations} ---")
        
        all_island_fitnesses = []
        
        # Evolve each island independently
        for island_idx in range(args.num_islands):
            print(f"\nEvaluating Island {island_idx + 1}/{args.num_islands}...")
            population = islands[island_idx]
            
            # Evaluate fitness
            fitness_scores = []
            for i, ind in enumerate(population):
                ssims = []
                psnrs = []
                for p_idx, p in enumerate(prompts):
                    img = generate_image(pipe, helper, p, args.seed, args.steps, interval_seq=ind)
                    
                    mse = F.mse_loss(img, baseline_images[p_idx]).item()
                    # Calculate PSNR
                    if mse == 0:
                        psnr = 100.0
                    else:
                        psnr = -10.0 * np.log10(mse)
                    psnrs.append(psnr)
                    
                    # Calculate SSIM
                    # img shape: [C, H, W], tensor values in [0, 1]
                    img_np = img.cpu().float().numpy()
                    base_np = baseline_images[p_idx].cpu().float().numpy()
                    
                    ssim_score = ssim_fn(base_np, img_np, data_range=1.0, channel_axis=0)
                    ssims.append(ssim_score)
                    
                avg_ssim = sum(ssims) / len(ssims)
                avg_psnr = sum(psnrs) / len(psnrs)
                eval_ratio = len(ind) / float(args.steps)
                
                fit = (avg_ssim * args.weight_ssim) + (avg_psnr * args.weight_psnr) - (eval_ratio * args.weight_eval)
                
                fitness_scores.append(fit)
                print(f"Island {island_idx + 1} - Ind {i}: fitness={fit:.4f}, SSIM={avg_ssim:.4f}, PSNR={avg_psnr:.2f}, Evals={len(ind)}/{args.steps}")
                
                if fit > best_overall_fitness:
                    best_overall_fitness = fit
                    best_overall_individual = ind
                    best_overall_ssim = avg_ssim
                    best_overall_psnr = avg_psnr
                    best_overall_evals = len(ind)
            
            all_island_fitnesses.append(fitness_scores)
            
            # Selection (Tournament)
            new_population = []
            # Elitism: keep best 2
            sorted_indices = np.argsort(fitness_scores)[::-1]
            new_population.append(population[sorted_indices[0]])
            if args.pop_size > 1:
                new_population.append(population[sorted_indices[1]])
            
            while len(new_population) < args.pop_size:
                # Tournament selection
                t1 = random.randint(0, args.pop_size - 1)
                t2 = random.randint(0, args.pop_size - 1)
                p1 = population[t1] if fitness_scores[t1] > fitness_scores[t2] else population[t2]
                
                t1 = random.randint(0, args.pop_size - 1)
                t2 = random.randint(0, args.pop_size - 1)
                p2 = population[t1] if fitness_scores[t1] > fitness_scores[t2] else population[t2]
                
                child = crossover(p1, p2, args.steps)
                child = mutate(child, args.steps, mutation_rate=0.1)
                new_population.append(child)
                
            islands[island_idx] = new_population

        # Migration logic
        if gen > 0 and (gen + 1) % args.migration_interval == 0 and args.num_islands > 1:
            print("\n--- Migration Triggered ---")
            migrants = []
            # Extract the best individuals from each island to migrate
            for island_idx in range(args.num_islands):
                fitness_scores = all_island_fitnesses[island_idx]
                sorted_indices = np.argsort(fitness_scores)[::-1]
                island_migrants = [islands[island_idx][idx] for idx in sorted_indices[:args.migration_size]]
                migrants.append(island_migrants)
            
            # Ring topology: Island i sends to Island (i+1)%N
            for island_idx in range(args.num_islands):
                next_island = (island_idx + 1) % args.num_islands
                next_fitness_scores = all_island_fitnesses[next_island]
                
                # Replace the worst individuals in the target island
                worst_indices = np.argsort(next_fitness_scores)
                for i in range(args.migration_size):
                    replace_idx = worst_indices[i]
                    islands[next_island][replace_idx] = migrants[island_idx][i]
            
            print(f"Migrated {args.migration_size} individual(s) between sub-populations (Ring Topology).")

    print("\n--- GA Search Complete ---")
    print(f"Best Cache Schedule (Evaluation Steps): {best_overall_individual}")
    print(f"Total Evaluations: {best_overall_evals}/{args.steps}")
    print(f"SSIM against Baseline: {best_overall_ssim:.4f}")
    print(f"PSNR against Baseline: {best_overall_psnr:.2f} dB")
    print(f"Fitness Score: {best_overall_fitness:.4f}")
    
    # Save the best image (using the first prompt)
    img = generate_image(pipe, helper, prompts[0], args.seed, args.steps, interval_seq=best_overall_individual)
    from torchvision.utils import save_image
    save_image(img, "best_ga_schedule.png")
    save_image(baseline_images[0], "baseline.png")
    
    if args.eval_coco:
        print("\n--- Running Automatic COCO Evaluation ---")
        import subprocess
        import os
        import glob
        
        # 1. Format sequence
        seq_str = ",".join(map(str, best_overall_individual))
        
        # 2. Run generate.py
        gen_cmd = [
            "python", "experiments/generate.py",
            "--dataset", args.dataset,
            "--steps", str(args.steps),
            "--batch_size", "4",
            "--layer", "0",
            "--block", "0",
            "--interval_seq", seq_str,
            "--num_prompts", str(args.eval_prompts)
        ]
        print(f"Running: {' '.join(gen_cmd)}")
        try:
            subprocess.run(gen_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Generation failed: {e}")
            return
            
        # 3. Find newest generated .pt file
        pt_files = glob.glob(f"{args.dataset}_ckpt/images-ga-sequence-time-*.pt")
        if not pt_files:
            print(f"Could not find generated .pt file in {args.dataset}_ckpt/ directory.")
            return
            
        newest_pt = max(pt_files, key=os.path.getctime)
        print(f"Found generation output: {newest_pt}")
        
        # 4. Run clip score
        clip_cmd = ["python", "experiments/clip_score.py", newest_pt]
        print(f"Running: {' '.join(clip_cmd)}")
        subprocess.run(clip_cmd)

    if args.eval_fid:
        print("\n--- Running Automatic FID Evaluation ---")
        import subprocess
        
        seq_str = ",".join(map(str, best_overall_individual))
        
        fid_cmd = [
            "python", "/data1/deepanshi/calculate_fid_coco.py",
            "--dataset", args.dataset,
            "--num_images", str(args.fid_images),
            "--interval_seq", seq_str,
            "--steps", str(args.steps)
        ]
        
        print(f"Running: {' '.join(fid_cmd)}")
        try:
            subprocess.run(fid_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"FID Evaluation failed: {e}")

if __name__ == "__main__":
    main()
