from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_curves(history: list[dict], path: str | Path, keys: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    for key in keys:
        values = [row[key] for row in history if key in row]
        if values:
            plt.plot(range(1, len(values) + 1), values, label=key)
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_dag(tokens: list[dict], edges: list[tuple[int, int]], path: str | Path) -> None:
    graph = nx.DiGraph()
    colors = []
    for idx, token in enumerate(tokens):
        label = f"{idx}:{token['token_type']}\\nd={token['depth']}"
        graph.add_node(idx, label=label)
        colors.append("#66c2a5" if token["token_type"] == "paper" else "#fc8d62")
    graph.add_edges_from(edges)
    pos = nx.spring_layout(graph, seed=0)
    labels = nx.get_node_attributes(graph, "label")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    nx.draw(graph, pos, labels=labels, node_color=colors, node_size=1100, arrows=True, font_size=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_bar(names: list[str], values: list[float], path: str | Path, ylabel: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.bar(np.arange(len(names)), values)
    plt.xticks(np.arange(len(names)), names, rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
