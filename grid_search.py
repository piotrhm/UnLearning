import os
import itertools
import subprocess
import glob


# Parameter grids
guidance_scales = [9.0]
negative_guidances = [2.0]
lora_ranks = [4, 8, 16, 20]
lora_alphas = [8, 16]
lrs = [1e-4, 3e-5, 1e-5]
iterations = [200]
image_size = 512
ddim_steps = 50
ddim_eta = 0.0
seed = 42

config_path = "./configs/stable-diffusion/v1-inference.yaml"
ckpt_path = "/shared/results/common/gmhelm/models/sd-v1-4.ckpt"
device = "cuda:0"
target_modules = ["attn2.to_k", "attn2.to_v"]
output_dir = "output"
prompts_json = "data/cat.json"

grid = list(itertools.product(
    guidance_scales, negative_guidances, lora_ranks, lora_alphas, lrs, iterations
))

for (start_guidance, negative_guidance, lora_rank, lora_alpha, lr, iters) in grid:
    config = {
        "class_name": os.path.splitext(os.path.basename(prompts_json))[0],  
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "iterations": iters,
        "lr": lr,
        "start_guidance": start_guidance,
        "negative_guidance": negative_guidance,
        "ddim_steps": ddim_steps,
    }
    exp_name = "_".join(f"{k}_{config[k]}" for k in [
        "class_name",
        "lora_rank",
        "lora_alpha",
        "iterations",
        "lr",
        "start_guidance",
        "negative_guidance",
        "ddim_steps"
    ])
    
    exp_dir = os.path.join(output_dir, exp_name)
    model_dir = os.path.join(exp_dir, "models")
    images_dir = os.path.join(exp_dir, "images")
    metrics_dir = os.path.join(exp_dir, "metrics")

    print("Experiment Name:", exp_name)

    # 1. Train
    train_cmd = [
        "python", "train.py",
        "--config_path", config_path,
        "--ckpt_path", ckpt_path,
        "--device", device,
        "--lora_rank", str(lora_rank),
        "--lora_alpha", str(lora_alpha),
        "--target_modules", *target_modules,
        "--iterations", str(iters),
        "--lr", str(lr),
        "--image_size", str(image_size),
        "--seed", str(seed),
        "--ddim_steps", str(ddim_steps),
        "--ddim_eta", str(ddim_eta),
        "--start_guidance", str(start_guidance),
        "--negative_guidance", str(negative_guidance),
        "--output_dir", output_dir,
        "--prompts_json", prompts_json,
        "--save_losses"
    ]
    print("Running:", " ".join(train_cmd))
    subprocess.run(train_cmd, check=True)

    # 2. Generate images
    gen_cmd = [
        "python", "generate_images.py",
        "--config", config_path,
        "--ckpt", ckpt_path,
        "--output_dir", output_dir,
        "--samples", "10",
        "--steps", str(ddim_steps),
        "--w1", str(-1),
        "--w2", str(2),
        "--seed", str(seed),
        "--device", device,
        "--decide_w", False
    ]
    print("Running:", " ".join(gen_cmd))
    subprocess.run(gen_cmd, check=True)
    
    os.makedirs(metrics_dir, exist_ok=True)
    # Iterate over class_name directories inside samples_dir
    for class_path in glob.glob(os.path.join(images_dir, "*")):
        class_name = os.path.basename(class_path)
        metrics_output_dir = os.path.join(metrics_dir, class_name)
        os.makedirs(metrics_output_dir, exist_ok=True)
        cmd = [
            "python", "compute_metrics.py", "-m", "acc",
            "--prompts_json", prompts_json,
            "--samples_dir", os.path.join(images_dir, class_name),
            "--output_dir", metrics_output_dir
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    # 4. Merge metrics
    merge_cmd = [
        "python", "merge_metrics_results.py",
        "-m", "acc",
        "--results_dir", metrics_dir,
        "--prompts_json", prompts_json
    ]
    print("Running:", " ".join(merge_cmd))
    subprocess.run(merge_cmd, check=True)