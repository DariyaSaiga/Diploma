import torch
import torch.nn as nn


def compute_separation_loss(domain_data: dict, device='cpu') -> torch.Tensor:
    """Minimize cosine similarity between invariant and private domains."""
    losses = []
    for modality in ['text', 'audio', 'visual']:
        inv = domain_data.get(f'{modality}_invariant')
        priv = domain_data.get(f'{modality}_private')
        if inv is not None and priv is not None:
            cos_sim = torch.nn.functional.cosine_similarity(inv, priv, dim=1)
            losses.append(cos_sim.mean())

    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)


def compute_invariant_loss(domain_data: dict, labels: torch.Tensor, margin=1.0) -> torch.Tensor:
    """Pull same-emotion invariant representations, push different-emotion."""
    losses = []
    for modality in ['text', 'audio', 'visual']:
        inv = domain_data.get(f'{modality}_invariant')
        if inv is None:
            continue

        dists = torch.cdist(inv, inv)
        same_emotion = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        same_emotion.fill_diagonal_(0)

        same_dists = (dists * same_emotion).sum(dim=1) / same_emotion.sum(dim=1).clamp(min=1)

        diff_emotion = 1.0 - same_emotion
        diff_emotion.fill_diagonal_(0)
        diff_dists = (dists * diff_emotion).sum(dim=1) / diff_emotion.sum(dim=1).clamp(min=1)

        loss = torch.relu(same_dists - diff_dists + margin).mean()
        losses.append(loss)

    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=labels.device)


def compute_reconstruction_loss(recon_data: dict) -> torch.Tensor:
    """Reconstruct original features from [invariant || private]."""
    losses = []
    criterion = nn.MSELoss()
    for modality in ['text', 'audio', 'visual']:
        recon = recon_data.get(f'{modality}_recon')
        original = recon_data.get(f'{modality}_original')
        if recon is not None and original is not None:
            losses.append(criterion(recon, original))

    return torch.stack(losses).mean() if losses else torch.tensor(0.0)
