import torch
import torch.nn.functional as F


def forward_latent(self, x: torch.Tensor):
    A = self.A.to(x.device)
    B = self.B.to(x.device)
    L = self.default_lora_latent_mapping.to(x.device)
    
    #print("Inside forward_latent")
    #print(x @ A)
    #print(x @ A @ L.weight)
    #print(x @ A @ L.weight @ B)
    
    delta = x @ A @ L.weight @ B
    return delta * self.alpha
