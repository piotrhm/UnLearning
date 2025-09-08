import types

import torch
from peft.utils import _get_submodules
from tqdm import tqdm

from .latent_utils import forward_latent
from .svd_utils import svd_lowrank


def get_replacement_module(weight, type, reconstruct_config):
    cfg = reconstruct_config[type]
    if type != 'svd':
        raise NotImplementedError(f"{type} is currently not supported.")

    A, B = svd_lowrank(weight, rank=cfg['rank'], split_sigma=cfg.get('sigma_split', 'left'))

    final_enc = A.to(dtype=weight.dtype, device=weight.device).contiguous()
    final_dec = B.to(dtype=weight.dtype, device=weight.device).contiguous()

    return final_enc, final_dec


def init_module_weights(target_module: torch.nn.Linear, sigma: float):
    # Initialize weights with Gaussian distribution
    torch.nn.init.normal_(target_module.weight, mean=0, std=sigma)
    if hasattr(target_module, "bias"):
        # Set bias to zeros
        if target_module.bias is not None:
            torch.nn.init.zeros_(target_module.bias)


def replace_module_weights(param: torch.nn.Parameter, new_weight: torch.Tensor):
    new_weight = new_weight.to(param.device, dtype=param.dtype)
    with torch.no_grad():
        param.data.copy_(new_weight)


def find_and_initialize(model, peft_config, adapter_name, reconstr_type, reconstruct_config, writer):
    """
    :param adapter_name: options: 'default'
    :param reconstr_type: options: 'svd'
    """
    reconstruction_mode = reconstruct_config['reconstr_mode']
    lora_config = peft_config[adapter_name]
    
    is_target_modules_in_base_model = False
    key_list = [key for key, _ in model.named_modules()]
    assert (not isinstance(lora_config.target_modules, str))
    print("Iterating through model's specified modules to initialize A/B matrices.")
    for key in tqdm(key_list):
        target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
        if target_module_found:
            if not is_target_modules_in_base_model:
                is_target_modules_in_base_model = True
            _, target, _ = _get_submodules(model, key)

            if reconstruction_mode == 'separated':
                A_svd, B_svd = get_replacement_module(
                    weight=target.original.weight.T,
                    type=reconstr_type,
                    reconstruct_config=reconstruct_config
                )

                replace_module_weights(target.lora.B, A_svd)
                replace_module_weights(target.lora.A, B_svd)

                target.lora.forward = types.MethodType(forward_latent, target.lora)
                
                target.lora.default_lora_latent_mapping = torch.nn.Linear(lora_config.r, lora_config.r, bias=False)
                init_module_weights(target.lora.default_lora_latent_mapping, sigma=0.00001)
                target.lora.default_lora_latent_mapping.to(target.lora.A.device)
                
                target.lora.default_lora_latent_mapping.weight.requires_grad = True
                target.lora.A.requires_grad = False
                target.lora.B.requires_grad = False

    if not is_target_modules_in_base_model:
        raise ValueError(
            f"Target modules {lora_config.target_modules} not found in the base model. "
            f"Please check the target modules and try again."
        )
