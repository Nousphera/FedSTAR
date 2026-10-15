"""Joint on-device self-training and whole-clip evaluation."""

import random
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from .audio import features, read_wave
from .models import regularization_loss
from .methods import confidence_threshold, self_training_loss


def seed_everything(seed, threads=1):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AudioRows(Dataset):
    def __init__(self, records, training=False, seed=0):
        self.records, self.training, self.seed = records, training, seed

    def __len__(self):
        return len(self.records)

    def __getitem__(self, key):
        index, draw = key if isinstance(key, tuple) else (key, 0)
        row = self.records[index]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, index, draw]))
        value = features(read_wave(row["path"]), self.training, rng)
        return torch.as_tensor(value, dtype=torch.float32), row["label"]


def balanced_batches(records, batch_size, rng):
    """One unlabeled pass per epoch, with equally sized labeled batches."""
    labeled = np.array([i for i, row in enumerate(records) if row["label"] >= 0])
    unlabeled = np.array([i for i, row in enumerate(records) if row["label"] < 0])
    if not len(labeled):
        raise ValueError("Every client needs labeled examples")
    if batch_size < 1 or (len(unlabeled) and (batch_size < 2 or batch_size % 2)):
        raise ValueError("Joint self-training requires an even batch size")
    order = rng.permutation(unlabeled if len(unlabeled) else labeled)
    size = batch_size // 2 if len(unlabeled) else batch_size
    lorder, cursor, draw = rng.permutation(labeled), 0, 0
    for start in range(0, len(order), size):
        chosen = order[start:start + size].tolist()
        if len(unlabeled):
            paired = []
            for _ in chosen:
                if cursor == len(lorder):
                    lorder, cursor = rng.permutation(labeled), 0
                paired.append(int(lorder[cursor]))
                cursor += 1
            chosen = paired + chosen
        yield [(int(index), draw + i) for i, index in enumerate(chosen)]
        draw += len(chosen)


def train_local(model, records, cfg, seed, round_number, optimizer_state=None):
    seed_everything(seed, cfg["torch-threads"])
    model.to(cfg["device"]).train()
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=cfg["learning-rate"], eps=1e-7)
    if optimizer_state:
        for key, value in optimizer_state.items():
            index, name = key.split("/", 1)
            parameter = parameters[int(index)]
            optimizer.state[parameter][name] = value.cpu() if name == "step" else value.to(cfg["device"])
    threshold = confidence_threshold(round_number - 1, max(1, cfg["num-rounds"] - 1),
                                     cfg["threshold-start"], cfg["threshold-end"])
    loss_sum, steps, accepted, unlabelled, correct, labeled_count = 0., 0, 0, 0, 0, 0
    for epoch in range(cfg["local-epochs"]):
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
        data = AudioRows(records, training=True, seed=seed + epoch)
        batches = DataLoader(data, batch_sampler=balanced_batches(records, cfg["batch-size"], rng),
                             num_workers=cfg["num-workers"], pin_memory=cfg["device"].startswith("cuda"))
        for x, labels in batches:
            x, labels = x.to(cfg["device"]), labels.to(cfg["device"])
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss, info = self_training_loss(logits, labels, threshold, cfg["temperature"], cfg["beta"])
            loss = loss + regularization_loss(model)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite local training loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError("Nonfinite local gradient")
            optimizer.step()
            mask = labels >= 0
            correct += int((logits[mask].argmax(1) == labels[mask]).sum())
            labeled_count += int(mask.sum())
            loss_sum += float(loss.detach())
            steps += 1
            accepted += info["accepted"]
            unlabelled += info["unlabeled"]
    state = {f"{i}/{name}": value.detach().cpu() for i, p in enumerate(parameters)
             for name, value in optimizer.state[p].items()}
    model.cpu()
    return {"loss": loss_sum / steps, "accuracy": correct / labeled_count,
            "threshold": threshold, "accepted": accepted, "unlabeled": unlabelled, "steps": steps}, state


def _collate_clips(rows):
    clips, labels = zip(*rows)
    return torch.cat(clips), [len(clip) for clip in clips], torch.tensor(labels)


@torch.inference_mode()
def evaluate(model, records, cfg):
    if not records:
        raise ValueError("Evaluation requires held-out recordings")
    model.to(cfg["device"]).eval()
    batches = DataLoader(AudioRows(records), batch_size=cfg["batch-size"], collate_fn=_collate_clips,
                         num_workers=cfg["num-workers"], pin_memory=cfg["device"].startswith("cuda"))
    total_loss, correct = 0., 0
    for segments, lengths, labels in batches:
        outputs = torch.cat([model(chunk.to(cfg["device"])).cpu()
                             for chunk in segments.split(cfg["batch-size"])])
        logits = torch.stack([clip.mean(0) for clip in outputs.split(lengths)])
        loss = F.cross_entropy(logits, labels, reduction="sum")
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite evaluation loss")
        total_loss += float(loss)
        correct += int((logits.argmax(1) == labels).sum())
    model.cpu()
    return {"loss": total_loss / len(records), "accuracy": correct / len(records)}
