import torch
from torch import nn
from typing import List, Tuple

class GeneratorLoss(nn.Module):
    def __init__(self, gan_type: str = 'hinge'):
        super().__init__()
        if gan_type not in ['hinge', 'lsgan']:
            raise ValueError(f"Unsupported GAN type: {gan_type}. Must be 'hinge' or 'lsgan'.")
        self.gan_type = gan_type

    def forward(self, disc_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        loss = torch.zeros(1, device=disc_outputs[0].device, dtype=disc_outputs[0].dtype)
        gen_losses = []
        for dg in disc_outputs:
            if self.gan_type == 'hinge':
                l = torch.mean(torch.clamp(1 - dg, min=0))
            elif self.gan_type == 'lsgan':
                l = torch.mean((dg - 1)**2)
            else:
                raise NotImplementedError(f"GAN type '{self.gan_type}' is not implemented.")
            
            gen_losses.append(l)
            loss += l

        return loss, gen_losses

class DiscriminatorLoss(nn.Module):
    def __init__(self, gan_type: str = 'hinge'):
        super().__init__()
        if gan_type not in ['hinge', 'lsgan']:
            raise ValueError(f"Unsupported GAN type: {gan_type}. Must be 'hinge' or 'lsgan'.")
        self.gan_type = gan_type

    def forward(
        self, disc_real_outputs: List[torch.Tensor], disc_generated_outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        loss = torch.zeros(1, device=disc_real_outputs[0].device, dtype=disc_real_outputs[0].dtype)
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            if self.gan_type == 'hinge':
                r_loss = torch.mean(torch.clamp(1 - dr, min=0))
                g_loss = torch.mean(torch.clamp(1 + dg, min=0))
            elif self.gan_type == 'lsgan':
                r_loss = torch.mean((dr - 1)**2)
                g_loss = torch.mean(dg**2)
            else:
                raise NotImplementedError(f"GAN type '{self.gan_type}' is not implemented.")

            loss += r_loss + g_loss
            r_losses.append(r_loss)
            g_losses.append(g_loss)

        return loss, r_losses, g_losses

class FeatureMatchingLoss(nn.Module):
    def forward(self, fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]]) -> torch.Tensor:
        loss = torch.zeros(1, device=fmap_r[0][0].device, dtype=fmap_r[0][0].dtype)
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return loss

if __name__ == '__main__':
    disc_real_outputs_mock = [torch.randn(4, 1, 1, 1), torch.randn(4, 1, 1, 1)]
    disc_generated_outputs_mock = [torch.randn(4, 1, 1, 1), torch.randn(4, 1, 1, 1)]

    print("--- Using Hinge GAN Loss ---")
    gen_loss_hinge = GeneratorLoss(gan_type='hinge')
    disc_loss_hinge = DiscriminatorLoss(gan_type='hinge')

    g_loss_val, _ = gen_loss_hinge(disc_generated_outputs_mock)
    d_loss_val, _, _ = disc_loss_hinge(disc_real_outputs_mock, disc_generated_outputs_mock)

    print(f"Generator Hinge Loss: {g_loss_val.item():.4f}")
    print(f"Discriminator Hinge Loss: {d_loss_val.item():.4f}")
    print("-" * 30)

    print("--- Using LSGAN Loss ---")
    gen_loss_lsgan = GeneratorLoss(gan_type='lsgan')
    disc_loss_lsgan = DiscriminatorLoss(gan_type='lsgan')

    g_loss_val, _ = gen_loss_lsgan(disc_generated_outputs_mock)
    d_loss_val, _, _ = disc_loss_lsgan(disc_real_outputs_mock, disc_generated_outputs_mock)

    print(f"Generator LSGAN Loss: {g_loss_val.item():.4f}")
    print(f"Discriminator LSGAN Loss: {d_loss_val.item():.4f}")
    print("-" * 30)
    
    print("--- Feature Matching Loss Example ---")
    fmap_r_mock = [[torch.randn(4, 64, 16, 16)] for _ in range(2)]
    fmap_g_mock = [[torch.randn(4, 64, 16, 16)] for _ in range(2)]
    
    feature_loss_fn = FeatureMatchingLoss()
    fm_loss = feature_loss_fn(fmap_r_mock, fmap_g_mock)
    print(f"Feature Matching Loss: {fm_loss.item():.4f}")
