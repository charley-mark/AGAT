import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import time
import random
from tqdm import tqdm
import kornia.augmentation as K
import torchattacks
from model import ResNet
from agatdataloader import load_cifar10_data_agat

# ------------------------- Reproducibility -------------------------
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)

# ------------------------- Device -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------- Config ---------------------------
EPOCHS = 1
BATCH_SIZE = 128
RESULTS_PATH = "results/agat_pgd_batchwise.json"
os.makedirs("results", exist_ok=True)

# k values for AGAT + Random-k baselines
K_LIST = [0.1, 0.2, 0.3, 0.5]

# ------------------------- GPU Augmentation (Kornia) -------------------------
# NOTE: Kornia expects tensors in BCHW on the SAME device (GPU here)
augmentation_gpu = K.AugmentationSequential(
    K.RandomCrop((32, 32), padding=4),
    K.RandomHorizontalFlip(p=0.5),
    data_keys=["input"]
).to(device)


# ----------------------- API Score (batch-wise, GPU) -----------------
def compute_api_batch(model, images):
    """
    API(x) = || f(x) - f(Aug(x)) ||_2 using logits.
    Fully GPU-based (Kornia augmentations).
    """
    was_training = model.training
    model.eval()

    # ensure augmentation is in train mode to keep randomness enabled
    augmentation_gpu.train()

    with torch.no_grad():
        logits_clean = model(images)
        aug_images = augmentation_gpu(images)          # GPU augmentation (batched)
        logits_aug = model(aug_images)
        api = torch.norm(logits_clean - logits_aug, dim=1)  # (B,)

    if was_training:
        model.train()
    return api


# ----------------------- Selection ------------------------
def select_indices(api_scores, k_frac, mode):
    """
    mode:
      - "full": select all
      - "agat": top-k by API
      - "random": random k
    Returns: (sel_idx, rest_idx)
    """
    B = api_scores.numel()

    if mode == "full":
        sel_idx = torch.arange(B, device=api_scores.device)
        rest_idx = torch.empty(0, dtype=torch.long, device=api_scores.device)
        return sel_idx, rest_idx

    k = max(int(k_frac * B), 1)

    if mode == "agat":
        sel_idx = torch.topk(api_scores, k, largest=True).indices
    elif mode == "random":
        sel_idx = torch.randperm(B, device=api_scores.device)[:k]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    mask = torch.ones(B, dtype=torch.bool, device=api_scores.device)
    mask[sel_idx] = False
    rest_idx = torch.nonzero(mask, as_tuple=False).view(-1)  # SAFE (no squeeze)
    return sel_idx, rest_idx


# ----------------------- Evaluation -------------------------
def evaluate(model, data_loader, attack=None):
    model.eval()
    correct, total = 0, 0

    for batch in data_loader:
        images, labels = batch[0].to(device), batch[1].to(device)

        if attack is not None:
            if hasattr(attack, "set_model"):
                attack.set_model(model)
            images = attack(images, labels)

        with torch.no_grad():
            out = model(images)
            pred = out.argmax(dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    return 100.0 * correct / total


# ----------------------- One epoch train -------------------------
def train_one_epoch(model, train_loader, optimizer, attack_train, mode, k_frac, epoch, debug=False):
    """
    mode: "full" | "agat" | "random"
    """
    model.train()
    ce = nn.CrossEntropyLoss()
    t0 = time.time()

    # augmentation randomness needed during training for the "rest" subset too
    augmentation_gpu.train()

    for b, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} | {mode.upper()} | k={k_frac:.2f}")):
        images, labels = batch[0].to(device), batch[1].to(device)

        # 1) compute API scores for batch (GPU)
        api = compute_api_batch(model, images)

        # 2) select subset
        sel_idx, rest_idx = select_indices(api, k_frac, mode)

        parts_images, parts_labels = [], []

        # 3) PGD on selected subset
        if sel_idx.numel() > 0:
            if hasattr(attack_train, "set_model"):
                attack_train.set_model(model)
            adv_images = attack_train(images[sel_idx], labels[sel_idx])
            parts_images.append(adv_images)
            parts_labels.append(labels[sel_idx])

        # 4) GPU augmentation on remaining subset
        if rest_idx.numel() > 0:
            aug_rest = augmentation_gpu(images[rest_idx])
            parts_images.append(aug_rest)
            parts_labels.append(labels[rest_idx])

        if len(parts_images) == 0:
            continue

        x = torch.cat(parts_images, dim=0)
        y = torch.cat(parts_labels, dim=0)

        out = model(x)
        loss = ce(out, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if debug and b < 3:
            k_here = max(int(k_frac * images.size(0)), 1) if mode != "full" else images.size(0)
            print(
                f"[DEBUG] B={images.size(0)} k={k_here} sel={sel_idx.numel()} rest={rest_idx.numel()} "
                f"x={tuple(x.shape)} y={tuple(y.shape)} loss={loss.item():.4f}"
            )

    return time.time() - t0


# ------------------------- Main Loop -------------------------
def run():
    train_loader, _, test_loader = load_cifar10_data_agat(
        batch_size=BATCH_SIZE,
        worker_init_fn=seed_worker,
        generator=g
    )

    run_plan = [("full", 1.0)]
    for k in K_LIST:
        run_plan.append(("random", k))
        run_plan.append(("agat", k))

    results = {
        "config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "k_list": K_LIST,
            "seed": seed,
            "train_attack": {"type": "PGD", "eps": 8/255, "alpha": 2/255, "steps": 10},
            "eval_attack":  {"type": "PGD", "eps": 8/255, "alpha": 2/255, "steps": 50},
            "api": "||logits(x)-logits(aug(x))||_2",
            "augmentation": "Kornia RandomCrop+RandomHFlip (GPU)"
        },
        "runs": []
    }

    for mode, k_frac in run_plan:
        print(f"\n=== RUN: mode={mode} | k={k_frac} ===")

        model = ResNet(num_classes=10).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        attack_train = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10)
        attack_eval  = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=50)

        epoch_times = []
        for epoch in range(EPOCHS):
            dt = train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                attack_train=attack_train,
                mode=mode,
                k_frac=k_frac,
                epoch=epoch,
                debug=(epoch == 0)
            )
            epoch_times.append(dt)

        avg_time = float(np.mean(epoch_times))
        clean_acc = evaluate(model, test_loader, attack=None)
        robust_acc = evaluate(model, test_loader, attack=attack_eval)

        # diagnostic: mean API on test set
        model.eval()
        augmentation_gpu.train()
        api_vals = []
        for batch in test_loader:
            x = batch[0].to(device)
            api_vals.extend(compute_api_batch(model, x).detach().cpu().numpy().tolist())
        mean_api = float(np.mean(api_vals))

        row = {
            "mode": mode,
            "k": float(k_frac),
            "clean_acc": float(clean_acc),
            "robust_acc_pgd50": float(robust_acc),
            "mean_api_test": mean_api,
            "avg_time_per_epoch_sec": avg_time
        }
        results["runs"].append(row)

        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

        print(
            f"RESULT | mode={mode} k={k_frac:.2f} | "
            f"Clean={clean_acc:.2f}% | Robust(PGD-50)={robust_acc:.2f}% | "
            f"MeanAPI={mean_api:.4f} | Time/Epoch={avg_time:.2f}s"
        )

    print(f"\nSaved results to: {RESULTS_PATH}")


if __name__ == "__main__":
    run()