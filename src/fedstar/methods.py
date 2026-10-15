"""Paper self-training objective and deterministic client ownership."""

import math
import numpy as np
import torch
from torch.nn import functional as F


def confidence_threshold(completed, rounds, start=.5, end=.9):
    if rounds < 1 or not 0 <= completed <= rounds or not 0 <= start <= end <= 1:
        raise ValueError("Invalid confidence schedule")
    return start + (end - start) * (1 - math.cos(math.pi * completed / rounds)) / 2


def self_training_loss(logits, labels, threshold=.5, temperature=4., beta=.5):
    """Equation 7; temperature selects targets but never rescales training logits."""
    labeled = labels >= 0
    supervised = F.cross_entropy(logits[labeled], labels[labeled]) if labeled.any() else logits.sum() * 0
    unlabeled = logits[~labeled]
    if len(unlabeled):
        probabilities = (unlabeled.detach() / temperature).softmax(-1)
        confidence, targets = probabilities.max(-1)
        selected = confidence >= threshold
        unsupervised = (F.cross_entropy(unlabeled, targets, reduction="none") * selected).mean()
    else:
        selected = torch.empty(0, device=logits.device, dtype=torch.bool)
        unsupervised = logits.sum() * 0
    return supervised + beta * unsupervised, {
        "labeled-loss": float(supervised.detach()), "unlabeled-loss": float(unsupervised.detach()),
        "accepted": int(selected.sum()), "unlabeled": len(unlabeled),
    }


def _random_shards(rows, clients, variation, rng, *, require_nonempty=True):
    if len(rows) < clients and require_nonempty:
        raise ValueError("Increase the labeled fraction or reduce nr-clients: every client needs a label")
    mean = len(rows) / clients
    low = max(int(require_nonempty), math.floor(mean * (1 - variation)))
    high = max(low, math.ceil(mean * (1 + variation)))
    counts = rng.integers(low, high + 1, clients)
    while counts.sum() != len(rows):
        sign = 1 if counts.sum() < len(rows) else -1
        eligible = np.flatnonzero(counts < high if sign > 0 else counts > low)
        counts[rng.choice(eligible)] += sign
    order = rng.permutation(len(rows))
    return [[dict(rows[i]) for i in group] for group in np.split(order, np.cumsum(counts)[:-1])]


def _class_shards(rows, clients, mean, variation, rng):
    classes = np.unique([r["label"] for r in rows])
    if not 1 <= mean <= len(classes) or clients * mean < len(classes):
        raise ValueError("labeled-classes must permit coverage of every class across clients")
    low, high = max(1., mean * (1 - variation)), min(float(len(classes)), mean * (1 + variation))
    values = rng.uniform(low, high, clients)
    counts = np.floor(values).astype(int) + (rng.random(clients) < values % 1)
    target = round(clients * mean)
    while counts.sum() != target:
        sign = 1 if counts.sum() < target else -1
        eligible = np.flatnonzero(counts < math.ceil(high) if sign > 0 else counts > math.floor(low))
        counts[rng.choice(eligible)] += sign
    available = {int(k): sum(r["label"] == k for r in rows) for k in classes}
    if target > len(rows):
        raise ValueError("Too few labels for the requested per-client class coverage")
    for _ in range(1000):
        ownership = [set(rng.choice(classes, size=int(n), replace=False).tolist()) for n in counts]
        owners = {int(k): [i for i, labels in enumerate(ownership) if k in labels] for k in classes}
        if all(0 < len(owners[k]) <= available[k] for k in owners):
            break
    else:
        owners = _capacity_assignment(counts, classes, available, rng)
    shards = [[] for _ in range(clients)]
    for label, ids in owners.items():
        group = [r for r in rows if r["label"] == label]
        for cid, selection in zip(ids, np.array_split(rng.permutation(len(group)), len(ids))):
            shards[cid].extend(dict(group[i]) for i in selection)
    for shard in shards:
        rng.shuffle(shard)
    return shards


def _capacity_assignment(counts, classes, available, rng):
    """Resolve tight label budgets with a capacity-constrained bipartite flow.

    Every client gets its requested distinct classes; every class has at least
    one owner and no more owners than available labeled examples. This fallback
    avoids rejecting feasible allocations when random proposals miss them.
    """
    from scipy.sparse import lil_matrix
    from scipy.sparse.csgraph import maximum_flow

    clients, labels = rng.permutation(len(counts)), rng.permutation(classes)
    total = int(sum(counts))
    # Nodes: source, sink, demand source/sink, clients, classes.
    graph = lil_matrix((4 + len(clients) + len(labels),) * 2, dtype=np.int64)
    graph[2, 0] = total
    graph[1, 3] = total - len(labels)
    for i, cid in enumerate(clients):
        graph[0, 4 + i] = int(counts[cid])
        for j in range(len(labels)):
            graph[4 + i, 4 + len(clients) + j] = 1
    for j, label in enumerate(labels):
        node = 4 + len(clients) + j
        graph[node, 1] = min(available[int(label)], len(clients)) - 1
        graph[node, 3] = 1  # One mandatory owner per class.
    result = maximum_flow(graph.tocsr(), 2, 3)
    if result.flow_value != total:
        raise ValueError("Insufficient labels for the requested per-client class counts")
    return {int(label): [int(cid) for i, cid in enumerate(clients)
                        if result.flow[4 + i, 4 + len(clients) + j] > 0]
            for j, label in enumerate(labels)}


def partition_records(records, cfg):
    """Class-proportional labels; independent unlabeled RNG keeps comparisons matched."""
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("Training records must be nonempty and uniquely identified")
    rng = np.random.default_rng(cfg["seed"])
    labeled, unlabeled = [], []
    for label in sorted({r["label"] for r in records}):
        if label < 0:
            raise ValueError("Prepared downstream data must have valid class labels")
        group = [r for r in records if r["label"] == label]
        order = rng.permutation(len(group))
        count = int(len(group) * cfg["labeled-fraction"])
        labeled.extend({k: r[k] for k in ("id", "path", "label")} for r in (group[i] for i in order[:count]))
        unlabeled.extend({"id": group[i]["id"], "path": group[i]["path"], "label": -1} for i in order[count:])
    clients = cfg["num-clients"]
    if cfg["method"] == "centralized":
        if not labeled:
            raise ValueError("The selected labeled fraction contains no examples")
        return [labeled]
    if cfg["labeled-classes"]:
        if {r["label"] for r in labeled} != {r["label"] for r in records}:
            raise ValueError("Class-skew allocation needs at least one label for every class")
        shards = _class_shards(labeled, clients, cfg["labeled-classes"], cfg["class-variation"], rng)
    else:
        shards = _random_shards(labeled, clients, cfg["quantity-variation"], rng)
    if cfg["method"] == "fedstar" and unlabeled and cfg["unlabeled-fraction"] > 0:
        urandom = np.random.default_rng(np.random.SeedSequence([cfg["seed"], 1]))
        count = int(len(unlabeled) * cfg["unlabeled-fraction"])
        subset = [unlabeled[i] for i in urandom.permutation(len(unlabeled))[:count]]
        other = _random_shards(subset, clients, cfg["quantity-variation"], urandom, require_nonempty=False)
        shards = [a + b for a, b in zip(shards, other)]
    return shards


def aggregate(states, weights, reference):
    if not states or len(states) != len(weights):
        raise ValueError("Missing client updates")
    weights = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Positive finite client sample counts are required")
    for state in states:
        if set(state) != set(reference):
            raise ValueError("Client model keys differ")
        for key, value in state.items():
            if value.shape != reference[key].shape or value.dtype != reference[key].dtype or not torch.isfinite(value).all():
                raise ValueError(f"Invalid client tensor: {key}")
    weights /= weights.sum()
    return {key: sum(state[key].double() * float(w) for state, w in zip(states, weights)).to(value.dtype)
            for key, value in reference.items()}
