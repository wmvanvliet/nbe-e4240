"""Run a stimulation with the predictive coding model on the GPU and plot the results.

This script reproduces Figures 5 and 6A of:

Nour Eddine, Samer, Trevor Brothers, Lin Wang, Michael Spratling, and Gina R. Kuperberg.
"A Predictive Coding Model of the N400". bioRxiv, 11 April 2023.
https://doi.org/10.1101/2023.04.10.536279.
"""

import numpy as np
import torch
from IPython.display import HTML
from matplotlib import animation
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm, trange

from n400_model import N400Model  # GPU model
from weights import get_weights

# Make sure to set this to where you've downloaded Samer's data package to.
data_path = "./data"

# You can play with these to run more or fewer simulation steps
n_pre_iterations = 2
n_iterations = 20
batch_size = 512

# Instantiate the model
weights = get_weights(data_path)
m = N400Model(weights, batch_size=batch_size).cpu()
torch.set_num_threads(1)
init_state = m.state_dict()

# # Grab the list of words in the experiment. We will use only the first 512 as inputs.
# with open(f"{data_path}/1579words_words.txt") as f:
#     lex = f.read().strip().split("\n")
# input_batch = lex[:batch_size]  # only use the first 512 words to test the model
#
# lex_sem_state = list()
# lex_sem_reconstruction = list()
# lex_sem_prederr = list()
# n_steps = 1000
# for i in range(n_steps):
#     if i < n_pre_iterations:
#         m("zeros")
#     else:
#         m(input_batch)
#     lex_sem_state.append(
#         (
#             m.layers.lex.state.detach().sum(axis=1)
#             + m.layers.sem.state.detach().sum(axis=1)
#         )
#         .mean(axis=0)
#         .cpu()
#     )
#     lex_sem_reconstruction.append(
#         (
#             m.layers.lex.reconstruction.detach().sum(axis=1)
#             + m.layers.sem.reconstruction.detach().sum(axis=1)
#         )
#         .mean(axis=0)
#         .cpu()
#     )
#     lex_sem_prederr.append(
#         (
#             m.layers.lex.bu_err.detach().sum(axis=1)
#             + m.layers.sem.bu_err.detach().sum(axis=1)
#         )
#         .mean(axis=0)
#         .cpu()
#     )
# lex_sem_state = np.array(lex_sem_state)
# lex_sem_reconstruction = np.array(lex_sem_reconstruction)
# lex_sem_prederr = np.array(lex_sem_prederr)

store = np.load("data/precomputed_lex_sem_512_words.npz")

with open(f"{data_path}/pseudowords.txt") as f:
    pseudowords = f.read().strip().split("\n")


def _get_data(m, kind="state"):
    if kind == "state":
        orth_data = m.layers.orth.state.detach().cpu()
        lex_data = m.layers.lex.state.detach().cpu()
        sem_data = m.layers.sem.state.detach().cpu()
    elif kind == "reconstruction":
        orth_data = m.layers.orth.reconstruction.detach().cpu()
        lex_data = m.layers.lex.reconstruction.detach().cpu()
        sem_data = m.layers.sem.reconstruction.detach().cpu()
    elif kind == "prederr":
        orth_data = m.layers.orth.bu_err.detach().cpu()
        lex_data = m.layers.lex.bu_err.detach().cpu()
        sem_data = m.layers.sem.bu_err.detach().cpu()
    return orth_data, lex_data, sem_data


def run_model_batch(words, n_steps=22, plot="state"):
    """Run the model on the given words."""
    if not isinstance(words, list):
        raise ValueError("The parameter `word` should be a list of strings.")
    if len(words) < 1:
        raise ValueError("`words` list is empty.")
    if plot not in ["state", "reconstruction", "prederr"]:
        raise ValueError(
            "`plot` should be one of: 'state', 'reconstruction', 'prederr'"
        )
    m.reset(batch_size=len(words))
    data = list()
    for i in trange(n_steps):
        if i < n_pre_iterations:
            m("zeros")
        else:
            m(words)
        _, lex_data, sem_data = _get_data(m, kind=plot)
        data.append((lex_data.sum(axis=1) + sem_data.sum(axis=1)).mean(axis=0))
    data = np.array(data)
    fig = plt.figure(figsize=(6, 4))
    plt.plot(store[plot][:n_steps], label="baseline words")
    plt.plot(data, label="selected words")
    plt.xlabel("steps")
    plt.ylabel(f"lexico-semantic {plot}")
    plt.legend(loc="upper right")
    plt.tight_layout()
    return fig


def run_model(word, n_steps=22, plot="state"):
    """Run the model on a given word."""
    if not isinstance(word, str):
        raise ValueError("The parameter `word` should be a single string.")
    if plot not in ["state", "reconstruction", "prederr"]:
        raise ValueError(
            "`plot` should be one of: 'state', 'reconstruction', 'prederr'"
        )
    fig, axes = plt.subplots(
        ncols=3, width_ratios=[0.3, 1, 1], figsize=(11, 5), layout="constrained"
    )

    img_orth = axes[0].imshow(np.zeros((26, 4)), vmin=0, vmax=2, cmap="RdBu_r")
    axes[0].set_yticks(np.arange(26), list("abcdefghijklmnopqrstuvwxyz"))
    axes[0].set_xticks(np.arange(4), ["1", "2", "3", "4"])
    axes[0].set_title("Orth. layer")

    img_lex = axes[1].imshow(
        np.zeros((40, 40)), vmin=0, vmax=2, cmap="RdBu_r", extent=[0, 40, 0, 40]
    )
    axes[1].set_axis_off()
    axes[1].set_title("Lexical layer")

    img_sem = axes[2].imshow(
        np.zeros((114, 114)), vmin=0, vmax=2, cmap="RdBu_r", extent=[0, 114, 0, 114]
    )
    axes[2].set_axis_off()
    axes[2].set_title("Semantic layer")

    progress = tqdm(total=n_steps + 1)

    def init():
        m.reset(batch_size=1)

    def animate(i):
        if i == 0:
            m.reset(batch_size=1)
        if i < n_pre_iterations:
            m("zeros")
        else:
            m([word])
        progress.update(1)
        orth_data, lex_data, sem_data = _get_data(m, kind=plot)
        img_orth.set_data(orth_data.reshape(4, 26).T)

        lex_grid = np.zeros((40 * 40))
        lex_grid[: len(lex_data.ravel())] = lex_data.ravel()
        lex_grid = lex_grid.reshape(40, 40)
        img_lex.set_data(lex_grid)

        sem_grid = np.zeros((114 * 114))
        sem_grid[: len(sem_data.ravel())] = sem_data.ravel()
        sem_grid = sem_grid.reshape(114, 114)
        img_sem.set_data(sem_grid)

        return img_orth, img_lex, img_sem

    anim = animation.FuncAnimation(
        fig, animate, frames=n_steps, interval=100, blit=False, repeat=True
    )
    return anim
