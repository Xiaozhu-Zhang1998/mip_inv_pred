"""Baseline implementation from the supplied research notebook. See docs/baselines.md."""
import numpy as np

# From original notebook, cell 69
import numpy as np

def irm_linear(
    X_list, y_list,
    penalty_weight=1e4,
    penalty_anneal_iters=100,
    l2_weight=1e-3,
    lr=1e-3,
    steps=500,
    device=None
):
    import torch
    from torch import nn,optim,autograd
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    X = [torch.tensor(x, dtype=torch.float32, device=device) for x in X_list]
    y = [torch.tensor(v, dtype=torch.float32, device=device).view(-1, 1) for v in y_list]

    p = X[0].shape[1]
    model = nn.Linear(p, 1, bias=False).to(device)
    nn.init.xavier_uniform_(model.weight)

    def penalty(pred, target):
        scale = torch.tensor(1., device=device, requires_grad=True)
        loss = ((pred * scale - target) ** 2).mean()
        grad = autograd.grad(loss, scale, create_graph=True)[0]
        return grad.pow(2)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for step in range(steps):
        risks, penalties = [], []

        for Xe, ye in zip(X, y):
            pred = model(Xe)
            risks.append(((pred - ye) ** 2).mean())
            penalties.append(penalty(pred, ye))

        risk = torch.stack(risks).mean()
        irm_penalty = torch.stack(penalties).mean()
        l2 = sum(w.pow(2).sum() for w in model.parameters())

        pw = penalty_weight if step >= penalty_anneal_iters else 1.0
        loss = risk + l2_weight * l2 + pw * irm_penalty
        if pw > 1: loss /= pw

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model.weight.detach().cpu().numpy().reshape(-1)

