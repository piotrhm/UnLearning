import torch
import torch.nn.functional as F


def forward_latent(self, x: torch.Tensor):
    A = self.A.to(x.device)
    B = self.B.to(x.device)
    L = self.default_lora_latent_mapping.to(x.device)
    
    delta = A @ L.weight @ B
    delta = delta.T
    return delta * self.alpha
