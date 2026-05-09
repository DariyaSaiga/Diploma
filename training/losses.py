import torch
import torch.nn.functional as F


def compute_separation_loss(domain_data, device='cpu'):
    """Squared cosine similarity between invariant and private (should be ~0 = orthogonal)."""
    losses = []
    for mod in ['text', 'audio', 'visual']:
        inv = domain_data.get(f'{mod}_inv_pool')
        priv = domain_data.get(f'{mod}_priv_pool')
        if inv is not None and priv is not None:
            cos = F.cosine_similarity(inv, priv, dim=-1)
            losses.append((cos ** 2).mean())

    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def compute_invariant_loss(domain_data, labels=None):
    """Cross-modal invariant alignment: MSE between pooled invariants of different modalities."""
    inv_pools = []
    for mod in ['text', 'audio', 'visual']:
        inv = domain_data.get(f'{mod}_inv_pool')
        if inv is not None:
            inv_pools.append(inv)

    if len(inv_pools) < 2:
        dev = inv_pools[0].device if inv_pools else 'cpu'
        return torch.tensor(0.0, device=dev)

    loss = torch.tensor(0.0, device=inv_pools[0].device)
    count = 0
    for i in range(len(inv_pools)):
        for j in range(i + 1, len(inv_pools)):
            loss = loss + F.mse_loss(inv_pools[i], inv_pools[j])
            count += 1
    return loss / count


def compute_reconstruction_loss(recon_data):
    """MSE between reconstructed and original pooled features."""
    losses = []
    for mod in ['text', 'audio', 'visual']:
        recon = recon_data.get(f'{mod}_recon')
        orig = recon_data.get(f'{mod}_original')
        if recon is not None and orig is not None:
            losses.append(F.mse_loss(recon, orig))

    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()
