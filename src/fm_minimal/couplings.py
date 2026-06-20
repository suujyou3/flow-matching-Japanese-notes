import torch


def pairwise_squared_cost(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean transport cost between all source and target samples."""

    return torch.cdist(x0, x1, p=2).pow(2)


def random_coupling(
    x0: torch.Tensor,
    x1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pair each source sample with a random target sample in the same batch."""

    # Independent coupling in a finite batch: shuffle target samples.
    # 有限batchでのindependent couplingとして、target側をランダムに並べ替える。
    perm = torch.randperm(x1.shape[0], device=x1.device)
    return x0, x1[perm], perm


def greedy_minibatch_coupling(
    x0: torch.Tensor,
    x1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A tiny educational coupling that greedily matches nearby batch samples.

    This is not an exact optimal transport solver. It is intentionally dependency
    free, so the material can show where coupling enters the Flow Matching loss.
    """

    if x0.shape[0] != x1.shape[0]:
        raise ValueError("x0 and x1 must have the same batch size.")

    batch = x0.shape[0]
    cost = pairwise_squared_cost(x0, x1)
    remaining = torch.ones(batch, dtype=torch.bool, device=x0.device)
    perm = torch.empty(batch, dtype=torch.long, device=x0.device)

    # Match the samples whose nearest target is most unambiguous first.
    # 最も近いtargetがはっきりしているsourceから順に割り当てる。
    order = torch.argsort(cost.min(dim=1).values)
    for i in order.tolist():
        # Mask already-used target samples so each target is assigned once.
        # すでに使ったtargetをmaskし、各targetが1回だけ割り当てられるようにする。
        masked_cost = cost[i].masked_fill(~remaining, float("inf"))
        j = torch.argmin(masked_cost)
        perm[i] = j
        remaining[j] = False

    return x0, x1[perm], perm
