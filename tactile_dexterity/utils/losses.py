import torch
import torch.nn.functional as F

def l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(x, y)

def mse(x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x, y)

# VicREG losses
def compute_std_loss(rep, epsilon = 1e-04):
    rep = rep - rep.mean(dim = 0)
    rep_std = torch.sqrt(rep.var(dim = 0) + epsilon)
    return torch.mean(F.relu(1 - rep_std)) / 2.0

def off_diagonal(rep_cov):
    n, _ = rep_cov.shape
    return rep_cov.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def compute_cov_loss(rep, feature_size):
    rep_cov = (rep.T @ rep) / (rep.shape[0] - 1)
    return off_diagonal(rep_cov).pow_(2).sum().div(feature_size)

# Standard VICReg Loss
def vicreg_loss(input_rep, output_rep, feature_size, sim_coef, std_coef, cov_coef):
    sim_loss = F.mse_loss(input_rep, output_rep)
    std_loss = compute_std_loss(input_rep) + compute_std_loss(output_rep)
    cov_loss = compute_cov_loss(input_rep, feature_size) + compute_cov_loss(output_rep, feature_size)

    final_loss = (sim_coef * sim_loss) + (std_coef * std_loss) + (cov_coef * cov_loss)
    loss_dict = {
        'train_loss': final_loss.item(),
        'sim_loss': sim_loss.item(),
        'std_loss': std_loss.item(),
        'cov_loss': cov_loss.item()
    }

    return final_loss, loss_dict