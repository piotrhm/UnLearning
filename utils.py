from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import torch
import numpy as np

from ldm.models.diffusion.ddimcopy import DDIMSampler
from utils_xs_lora.svd_utils import svd_lowrank


def set_seed(seed: int):
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def load_model_from_config(config_path: str, ckpt_path: str, device: str = "cpu"):
    """Load and initialize model from configuration and checkpoint"""
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)

    # Load checkpoint weights
    pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(pl_sd["state_dict"], strict=False)

    # Move to device and set to eval mode
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device

    return model

def get_models(config_path: str, ckpt_path: str, device: str):
    """Initialize both original and trainable models with samplers"""
    # Original model (frozen, for reference)
    model_orig = load_model_from_config(config_path, ckpt_path, device)
    sampler_orig = DDIMSampler(model_orig)

    # Trainable model (will be modified with LoRA)
    model = load_model_from_config(config_path, ckpt_path, device)
    sampler = DDIMSampler(model)

    return model_orig, sampler_orig, model, sampler


def print_trainable_parameters(model, max_params: int = 10):
    """Print the first few trainable parameters"""
    print(f"First {max_params} layers with requires_grad == True:")
    count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            # print(f"  {name}")
            count += 1
            if count >= max_params:
                break

def apply_lora_to_model(model, lora_state_dict, alpha=4):
    """
    Apply LoRA adapters to a base model’s weights, scaling by the given alpha.

    Args:
        model (nn.Module): The base model containing original weights.
        lora_state_dict (dict): A state dict containing keys like
            "<prefix>.lora.A" and "<prefix>.lora.B" for each adapter.
        alpha (float): Scaling factor for the LoRA update (default: 4).
    """
    model_sd = model.state_dict()
        
    print("Applying LoRA adapters to model weights...")
    for lora_L_key in [k for k in lora_state_dict if k.endswith(".lora.default_lora_latent_mapping.weight")]:
        prefix = lora_L_key[:-len(".lora.default_lora_latent_mapping.weight")]
        print(prefix)
        
        L_key = prefix + ".lora.default_lora_latent_mapping.weight"
        W_key = prefix + ".weight"

        L = lora_state_dict[L_key].to(model_sd[W_key].device)
        print(L.shape)
        print(L)
        
        A, B = svd_lowrank(model_sd[W_key].T, rank=40, split_sigma='left')

        delta = A @ L @ B
        print(delta.shape)
        print(A.shape)
        print(B.shape)
        print(model_sd[W_key].shape)
        model_sd[W_key] = model_sd[W_key] + alpha * delta.T

    model.load_state_dict(model_sd, strict=False)
