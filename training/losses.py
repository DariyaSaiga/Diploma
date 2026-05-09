import torch
import torch.nn.functional as F


def compute_separation_loss(domain_data):
    """Squared cosine similarity between invariant and private (push toward orthogonal)."""
    losses = []
    for mod in ('text', 'audio', 'visual'):
        inv = domain_data.get(f'{mod}_inv_pool')
        priv = domain_data.get(f'{mod}_priv_pool')
        if inv is not None and priv is not None:
            cos = F.cosine_similarity(inv, priv, dim=-1)
            losses.append((cos ** 2).mean())

    if not losses:
        device = next(iter(domain_data.values())).device if domain_data else 'cpu'
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def compute_invariant_loss(domain_data):
    """Cross-modal alignment: MSE between invariant pools of different modalities."""
    inv_pools = [
        domain_data[k] for k in ('text_inv_pool', 'audio_inv_pool', 'visual_inv_pool')
        if k in domain_data
    ]

    if len(inv_pools) < 2:
        device = inv_pools[0].device if inv_pools else 'cpu'
        return torch.tensor(0.0, device=device)

    loss = torch.tensor(0.0, device=inv_pools[0].device)
    count = 0
    for i in range(len(inv_pools)):
        for j in range(i + 1, len(inv_pools)):
            loss = loss + F.mse_loss(inv_pools[i], inv_pools[j])
            count += 1
    return loss / count


def compute_reconstruction_loss(recon_data):
    """MSE between reconstructed features and original pooled features."""
    losses = []
    for mod in ('text', 'audio', 'visual'):
        recon = recon_data.get(f'{mod}_recon')
        orig = recon_data.get(f'{mod}_original')
        if recon is not None and orig is not None:
            losses.append(F.mse_loss(recon, orig))

    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


def compute_total_loss(logits, labels, domain_data, recon_data, criterion, args):
    """Combine task loss with optional domain separation auxiliary losses.

    Returns: (total_loss, loss_dict) where loss_dict contains individual loss values.
    """
    task_loss = criterion(logits, labels)
    loss_dict = {"task": task_loss.item()}

    if not (args.use_domain_sep and domain_data):
        return task_loss, loss_dict

    L_sep = compute_separation_loss(domain_data)
    L_inv = compute_invariant_loss(domain_data)
    L_rec = compute_reconstruction_loss(recon_data)

    total = (task_loss
             + args.alpha_sep * L_sep
             + args.alpha_inv * L_inv
             + args.alpha_rec * L_rec)

    loss_dict.update({
        "sep": L_sep.item(),
        "inv": L_inv.item(),
        "rec": L_rec.item(),
    })
    return total, loss_dict
