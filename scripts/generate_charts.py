#!/usr/bin/env python3
"""
Generate benchmark charts for RoBC documentation.

This script generates the performance comparison charts used in the README.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

ASSETS_DIR = Path(__file__).parent.parent / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

BENCHMARK_DATA = {
    "routers": [
        "RoBC",
        "RoRF (Static)",
        "RoRF (Retrain)",
        "ε-Greedy\n(Contextual)",
        "Always\nGeneralist",
        "ε-Greedy\n(Global)",
        "Random",
    ],
    "avg_reward": [0.9005, 0.8977, 0.8971, 0.8857, 0.8030, 0.7920, 0.6977],
    "std_reward": [0.0007, 0.0127, 0.0069, 0.0016, 0.0005, 0.0010, 0.0012],
    "specialist_pct": [91.0, 86.1, 92.0, 90.4, 15.0, 15.2, 16.9],
}

COLORS = {
    "robc": "#2563eb",
    "rorf": "#7c3aed", 
    "contextual": "#059669",
    "global": "#dc2626",
    "random": "#6b7280",
}

def get_color(router_name):
    name_lower = router_name.lower()
    if "robc" in name_lower:
        return COLORS["robc"]
    elif "rorf" in name_lower:
        return COLORS["rorf"]
    elif "contextual" in name_lower:
        return COLORS["contextual"]
    elif "global" in name_lower or "generalist" in name_lower:
        return COLORS["global"]
    else:
        return COLORS["random"]


def create_performance_chart():
    """Create the main performance comparison bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    routers = BENCHMARK_DATA["routers"]
    rewards = BENCHMARK_DATA["avg_reward"]
    stds = BENCHMARK_DATA["std_reward"]
    
    colors = [get_color(r) for r in routers]
    
    x = np.arange(len(routers))
    bars = ax.bar(x, rewards, yerr=stds, capsize=5, color=colors, edgecolor='white', linewidth=1.5)
    
    ax.set_ylabel("Average Quality Score", fontsize=12, fontweight='bold')
    ax.set_xlabel("Router Type", fontsize=12, fontweight='bold')
    ax.set_title("RoBC vs Other Routers: Quality Performance\n(50,000 evaluations across 5 random seeds)", 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(routers, fontsize=10)
    ax.set_ylim(0.65, 0.95)
    
    ax.axhline(y=rewards[0], color=COLORS["robc"], linestyle='--', alpha=0.3, linewidth=2)
    
    for i, (bar, reward) in enumerate(zip(bars, rewards)):
        height = bar.get_height()
        ax.annotate(f'{reward:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "performance_comparison.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {ASSETS_DIR / 'performance_comparison.png'}")


def create_specialist_chart():
    """Create the specialist selection accuracy chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    routers = BENCHMARK_DATA["routers"]
    specialist_pct = BENCHMARK_DATA["specialist_pct"]
    
    colors = [get_color(r) for r in routers]
    
    x = np.arange(len(routers))
    bars = ax.bar(x, specialist_pct, color=colors, edgecolor='white', linewidth=1.5)
    
    ax.set_ylabel("Specialist Selection Rate (%)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Router Type", fontsize=12, fontweight='bold')
    ax.set_title("Contextual Learning: Selecting the Right Specialist\n(Higher = Better at picking domain-specific models)", 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(routers, fontsize=10)
    ax.set_ylim(0, 100)
    
    for bar, pct in zip(bars, specialist_pct):
        height = bar.get_height()
        ax.annotate(f'{pct:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "specialist_selection.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {ASSETS_DIR / 'specialist_selection.png'}")


def create_improvement_chart():
    """Create the improvement comparison chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    comparisons = [
        "vs Global ε-Greedy",
        "vs Always Generalist",
        "vs Random",
    ]
    
    robc_score = 0.9005
    others = [0.7920, 0.8030, 0.6977]
    improvements = [(robc_score - other) / other * 100 for other in others]
    
    colors = [COLORS["global"], COLORS["global"], COLORS["random"]]
    
    x = np.arange(len(comparisons))
    bars = ax.barh(x, improvements, color=COLORS["robc"], edgecolor='white', linewidth=1.5, height=0.6)
    
    ax.set_xlabel("Improvement (%)", fontsize=12, fontweight='bold')
    ax.set_title("RoBC Performance Improvement Over Baselines", 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_yticks(x)
    ax.set_yticklabels(comparisons, fontsize=11)
    
    for bar, imp in zip(bars, improvements):
        width = bar.get_width()
        ax.annotate(f'+{imp:.1f}%',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center',
                    fontsize=12, fontweight='bold', color=COLORS["robc"])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(0, 35)
    
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "improvement.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {ASSETS_DIR / 'improvement.png'}")


def create_architecture_diagram():
    """Create a simple architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    def draw_box(x, y, w, h, text, color, fontsize=11):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor=color, edgecolor='white', linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='white',
                wrap=True)
    
    def draw_arrow(x1, y1, x2, y2, label=None):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#374151', lw=2))
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y + 0.3, label, ha='center', va='bottom',
                    fontsize=9, color='#374151')
    
    draw_box(0.5, 5.5, 2.5, 1.5, "Prompt\n(Text)", "#6b7280", 12)
    draw_arrow(3, 6.25, 4, 6.25, "embed")
    
    draw_box(4, 5.5, 2.5, 1.5, "Embedding\nVector", "#059669", 11)
    draw_arrow(6.5, 6.25, 7.5, 6.25)
    
    draw_box(7.5, 5.5, 3, 1.5, "Cluster\nManager\n(kNN)", "#7c3aed", 11)
    draw_arrow(10.5, 6.25, 11.5, 6.25)
    
    draw_box(11.5, 5.5, 2, 1.5, "Cluster\nWeights", "#7c3aed", 10)
    
    draw_arrow(12.5, 5.5, 12.5, 4.5)
    
    draw_box(7.5, 2.5, 3, 1.5, "Posterior\nManager\n(Bayesian)", "#2563eb", 11)
    draw_arrow(10.5, 3.25, 11.5, 3.25)
    
    draw_box(11.5, 2.5, 2, 1.5, "Model\nPosteriors", "#2563eb", 10)
    
    draw_arrow(12.5, 2.5, 12.5, 1.5)
    
    draw_box(10, 0, 3.5, 1.2, "Thompson Sampler", "#dc2626", 11)
    draw_arrow(10, 0.6, 8.5, 0.6)
    
    draw_box(5.5, 0, 3, 1.2, "Selected Model", "#059669", 11)
    
    draw_box(0.5, 2, 3.5, 2, "Feedback\nLoop\n(quality score)", "#f59e0b", 11)
    draw_arrow(4, 3, 7.5, 3.25, "update")
    
    ax.text(7, 7.5, "RoBC Architecture", ha='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "architecture.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {ASSETS_DIR / 'architecture.png'}")


if __name__ == "__main__":
    print("Generating RoBC benchmark charts...")
    create_performance_chart()
    create_specialist_chart()
    create_improvement_chart()
    create_architecture_diagram()
    print("\nAll charts generated successfully!")
