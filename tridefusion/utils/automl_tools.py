import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import optuna
from optuna.importance import get_param_importances
from pymfe.mfe import MFE
import copy
from ..train import train


def sample_random_params():
    return {
        "lr": 10 ** np.random.uniform(-5, -3),
        "batch_size": random.choice([4, 8, 12, 16, 32]),
        "wd": 10 ** np.random.uniform(-6, -2),
        "alpha": np.random.uniform(0.5, 1.5),
        "beta": np.random.uniform(0.1, 0.8),
        "delta": np.random.uniform(0.1, 0.8),
        "omega": np.random.uniform(0.0, 0.3),
    }


def optuna_objective(trial, base_args):
    args = copy.deepcopy(base_args)
    args.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    args.batch_size = trial.suggest_categorical("batch_size", [4, 8, 12, 16, 32])
    args.wd = trial.suggest_float("wd", 1e-6, 1e-2, log=True)

    alpha = trial.suggest_float("alpha", 0.5, 1.5)
    beta  = trial.suggest_float("beta", 0.1, 0.8)
    delta = trial.suggest_float("delta", 0.1, 0.8)
    omega = trial.suggest_float("omega", 0.0, 0.3)
    args.epochs = 20
    args.exp_name = f"optuna_trial_{trial.number}"
    args.loss_params = dict(
        alpha=alpha,
        beta=beta,
        delta=delta,
        omega=omega,
    )
    score = train(args)
    trial.report(score, step=0)
    params = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "wd": args.wd,
        **args.loss_params
    }
    print(f"Trial {trial.number}: PSNR={score:.4f}, params={params}")
    return score

def plot_optuna_param_importance(study):
    importances = get_param_importances(study)
    imp_df = pd.DataFrame({
        "parameter": list(importances.keys()),
        "importance": list(importances.values())
    }).sort_values("importance", ascending=True)
    plt.figure(figsize=(8, 5))
    plt.barh(imp_df["parameter"], imp_df["importance"])
    plt.title("Optuna Hyperparameter Importance (fANOVA)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

def image_stats(x: torch.Tensor):
    x = x.view(-1).numpy()
    return [
        x.mean(),
        x.std(),
        x.min(),
        x.max(),
        np.percentile(x, 25),
        np.percentile(x, 50),
        np.percentile(x, 75),
        np.mean(np.abs(x)),
        np.mean(x**2),
        np.var(x),
    ]


def extract_metafeatures(loader):
    meta_samples = []
    for _, (_, _, clean) in enumerate(loader):
        B = clean.shape[0]
        for i in range(B):
            stats = image_stats(clean[i])
            meta_samples.append(stats)

    meta_samples = np.array(meta_samples)
    print(meta_samples.shape)
    mfe = MFE(
        groups=["statistical", "info-theory", "complexity"],
        summary=["mean", "sd"]
    )
    mfe.fit(meta_samples)
    feature_names, feature_values = mfe.extract()
    meta_features = torch.tensor(feature_values, dtype=torch.float32).unsqueeze(0)

    print("Meta-features shape:", meta_features.shape)
    print("Meta-features names:")
    for n, v in zip(feature_names, feature_values):
        print(f"{n}: {v:.4f}")