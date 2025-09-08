import torch

def forward_latent(self, x: torch.Tensor):
    A = self.A.to(x.device)
    B = self.B.to(x.device)
    L = self.default_lora_latent_mapping.to(x.device)
    
    delta = x @ A @ L.weight @ B
    return delta * self.alpha
