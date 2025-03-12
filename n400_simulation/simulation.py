"""Run a stimulation with the predictive coding model and plot the results.

Nour Eddine, Samer, Trevor Brothers, Lin Wang, Michael Spratling, and Gina R. Kuperberg.
"A Predictive Coding Model of the N400". bioRxiv, 11 April 2023.
https://doi.org/10.1101/2023.04.10.536279.

Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
"""

import numpy as np
import torch
from IPython.display import HTML  # noqa
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

store = np.load("data/precomputed_lex_sem_512_words.npz")

with open(f"{data_path}/pseudowords.txt") as f:
    pseudowords = f.read().strip().split("\n")


def _get_data(m, kind="state"):
    if kind == "state":
        orth_data = m.layers.orth.state.detach().cpu()
        lex_data = m.layers.lex.state.detach().cpu()
        sem_data = m.layers.sem.state.detach().cpu()
    elif kind == "prediction":
        orth_data = m.layers.orth.reconstruction.detach().cpu()
        lex_data = m.layers.lex.reconstruction.detach().cpu()
        sem_data = m.layers.sem.reconstruction.detach().cpu()
    elif kind == "prederr":
        orth_data = m.layers.orth.bu_err.detach().cpu()
        lex_data = m.layers.lex.bu_err.detach().cpu()
        sem_data = m.layers.sem.bu_err.detach().cpu()
    return orth_data, lex_data, sem_data


def run_model_batch(words, n_steps=40, plot="state"):
    """Run the model on a batch of words.

    Parameters
    ----------
    words : list of str
        The words to run through the model.
    n_steps : int
        The number of steps to run the simulation for.
    plot : "state" | "prediction" | "prederr"
        What aspect of the model to plot.

    Returns
    -------
    fig : matplotlib.Figure
        The matplotlib figure.
    """
    if not isinstance(words, list):
        raise ValueError("The parameter `word` should be a list of strings.")
    if len(words) < 1:
        raise ValueError("`words` list is empty.")
    if plot not in ["state", "prediction", "prederr"]:
        raise ValueError("`plot` should be one of: 'state', 'prediction', 'prederr'")
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
    plt.plot(store[plot][:n_steps], label="real words")
    plt.plot(data, label="pseudowords")
    plt.xlabel("steps")
    plt.ylabel(f"lexico-semantic {plot}")
    plt.legend()
    plt.tight_layout()
    return fig


def run_model(word, n_steps=40, plot="state"):
    """Run the model on a given word.

    Parameters
    ----------
    word : str
        The word to run through the model.
    n_steps : int
        The number of steps to run the simulation for.
    plot : "state" | "prediction" | "prederr"
        What aspect of the model to plot.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The matplotlib animation.
    """
    if not isinstance(word, str):
        raise ValueError("The parameter `word` should be a single string.")
    if plot not in ["state", "prediction", "prederr"]:
        raise ValueError("`plot` should be one of: 'state', 'prediction', 'prederr'")
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


print("Everything is loaded, you're all good to go!")
