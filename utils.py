from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import torch
import numpy as np
import yaml

from ldm.models.diffusion.ddimcopy import DDIMSampler
from peft import LoraConfig


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
    
    from utils_xs_lora.initialization_utils import find_and_initialize
    adapter_name = "default"
    lora_config = LoraConfig(
            r=40,
            lora_alpha=16,
            target_modules=["attn2.to_k", "attn2.to_v"],
            lora_dropout=0,
            task_type="CAUSAL_LM",
        )
    peft_config_dict = {adapter_name: lora_config}

    with open("configs/reconstruct_config.yaml", 'r') as stream:
        reconstr_config = yaml.load(stream, Loader=yaml.FullLoader)
    reconstr_type = reconstr_config['reconstruction_type']
    reconstr_config[reconstr_type]['rank'] = peft_config_dict[adapter_name].r
    find_and_initialize(model, peft_config_dict, adapter_name=adapter_name, reconstr_type=reconstr_type,
                        writer=None, reconstruct_config=reconstr_config)
   
    for lora_L_key in [k for k in lora_state_dict if k.endswith(".lora.default_lora_latent_mapping")]:
        prefix = lora_L_key[:-len(".lora.default_lora_latent_mapping")]
        print(prefix)
        
        L_key = prefix + ".lora.default_lora_latent_mapping"
        W_key = prefix + ".weight"

        A = model.lora.A
        B = model.lora.B
        L = lora_state_dict[L_key].to(model_sd[W_key].device)

        delta = A @ L.weight @ B
        model_sd[W_key] = model_sd[W_key] + alpha * delta

    model.load_state_dict(model_sd, strict=False)
