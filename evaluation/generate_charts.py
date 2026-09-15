"""
Chart Generation Script for Experimental Results

This module generates publication-quality charts visualizing the experimental
comparison between the probability-based agent and the Q-learning RL agent.
Charts are saved in both PNG (for preview) and PDF (for paper inclusion) formats.

Charts Generated:
1. bar_chart_comparison - Average shots with standard deviation
2. box_plot - Distribution summary with quartiles and outliers
3. histogram - Overlapping frequency distributions with mean lines
4. accuracy_comparison - Hit accuracy comparison
5. cumulative_wins - Cumulative distribution function of games won
6. violin_plot - Kernel density distribution visualization

Input Files:
- results/probability_agent_results.csv
- results/rl_agent_results.csv

Output Directory: results/charts/

Author: Venkatashivasai Muppidi
Course: Foundations of Artificial Intelligence
University: Northeastern University
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# =============================================================================
# MATPLOTLIB CONFIGURATION
# =============================================================================

# Use seaborn-style whitegrid for clean academic appearance
plt.style.use('seaborn-v0_8-whitegrid')

# Font sizes for readability
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.figsize'] = (10, 6)

# =============================================================================
# DATA LOADING
# =============================================================================

# Create output directory for charts
os.makedirs('results/charts', exist_ok=True)

# Load experimental results
prob_df = pd.read_csv('results/probability_agent_results.csv')
rl_df = pd.read_csv('results/rl_agent_results.csv')

# Extract key metrics as numpy arrays
prob_shots = prob_df['shots'].values  # Shots to win per game
rl_shots = rl_df['shots'].values
prob_acc = prob_df['accuracy'].values  # Hit accuracy per game
rl_acc = rl_df['accuracy'].values


# =============================================================================
# CHART GENERATION FUNCTIONS
# =============================================================================


def create_bar_chart():
    """
    Create bar chart comparing average shots to win.

    Shows mean shots with error bars representing standard deviation.
    Includes value labels above each bar for clarity.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    agents = ['Probability\nAgent', 'RL Agent\n(Q-Learning)']
    avg_shots = [np.mean(prob_shots), np.mean(rl_shots)]
    std_shots = [np.std(prob_shots), np.std(rl_shots)]

    # Color scheme: green for probability, blue for RL
    colors = ['#2ecc71', '#3498db']
    bars = ax.bar(agents, avg_shots, yerr=std_shots, capsize=5, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels above bars
    for bar, avg in zip(bars, avg_shots):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f'{avg:.1f}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Average Shots to Win')
    ax.set_title('Agent Performance Comparison')
    ax.set_ylim(0, max(avg_shots) + max(std_shots) + 10)

    plt.tight_layout()
    plt.savefig('results/charts/bar_chart_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/charts/bar_chart_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("Created: bar_chart_comparison.png/pdf")


def create_box_plot():
    """
    Create box plot showing distribution of shots to win.

    Displays median, quartiles, and outliers for each agent.
    Useful for understanding distribution shape and variability.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    data = [prob_shots, rl_shots]
    labels = ['Probability Agent', 'RL Agent (Q-Learning)']
    colors = ['#2ecc71', '#3498db']

    bp = ax.boxplot(data, labels=labels, patch_artist=True)

    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Shots to Win')
    ax.set_title('Distribution of Shots to Win')

    plt.tight_layout()
    plt.savefig('results/charts/box_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/charts/box_plot.pdf', bbox_inches='tight')
    plt.close()
    print("Created: box_plot.png/pdf")


def create_histogram():
    """
    Create overlapping histogram of shots to win.

    Shows frequency distribution for both agents with vertical
    lines indicating mean values. Allows direct visual comparison
    of distribution overlap and separation.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bins from 15 to 100 in steps of 5
    bins = np.arange(15, 105, 5)

    # Plot overlapping histograms with transparency
    ax.hist(prob_shots, bins=bins, alpha=0.7, label='Probability Agent', color='#2ecc71', edgecolor='black')
    ax.hist(rl_shots, bins=bins, alpha=0.7, label='RL Agent (Q-Learning)', color='#3498db', edgecolor='black')

    # Add vertical mean lines
    ax.axvline(np.mean(prob_shots), color='#27ae60', linestyle='--', linewidth=2,
               label=f'Prob. Mean: {np.mean(prob_shots):.1f}')
    ax.axvline(np.mean(rl_shots), color='#2980b9', linestyle='--', linewidth=2,
               label=f'RL Mean: {np.mean(rl_shots):.1f}')

    ax.set_xlabel('Shots to Win')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Shots to Win per Game')
    ax.legend()

    plt.tight_layout()
    plt.savefig('results/charts/histogram.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/charts/histogram.pdf', bbox_inches='tight')
    plt.close()
    print("Created: histogram.png/pdf")


def create_accuracy_comparison():
    """
    Create bar chart comparing hit accuracy.

    Shows mean accuracy with error bars representing standard deviation.
    Accuracy = hits / total_shots * 100.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    agents = ['Probability\nAgent', 'RL Agent\n(Q-Learning)']
    avg_acc = [np.mean(prob_acc), np.mean(rl_acc)]
    std_acc = [np.std(prob_acc), np.std(rl_acc)]

    colors = ['#2ecc71', '#3498db']
    bars = ax.bar(agents, avg_acc, yerr=std_acc, capsize=5, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels with percentage
    for bar, avg in zip(bars, avg_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{avg:.1f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Agent Accuracy Comparison')
    ax.set_ylim(0, max(avg_acc) + max(std_acc) + 10)

    plt.tight_layout()
    plt.savefig('results/charts/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/charts/accuracy_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("Created: accuracy_comparison.png/pdf")


def create_cumulative_wins():
    """
    Create cumulative distribution function (CDF) of games won.

    Shows the percentage of games won as a function of shots used.
    Useful for understanding performance at different game lengths.
    Reference lines at 50% and 90% for quick comparison.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # X-axis: number of shots (17 is minimum possible: 5+4+3+3+2)
    shot_range = np.arange(17, 101)

    # Calculate cumulative percentage at each shot threshold
    prob_cumulative = [np.sum(prob_shots <= s) / len(prob_shots) * 100 for s in shot_range]
    rl_cumulative = [np.sum(rl_shots <= s) / len(rl_shots) * 100 for s in shot_range]

    ax.plot(shot_range, prob_cumulative, label='Probability Agent', color='#2ecc71', linewidth=2)
    ax.plot(shot_range, rl_cumulative, label='RL Agent (Q-Learning)', color='#3498db', linewidth=2)

    # Reference lines
    ax.axhline(50, color='gray', linestyle=':', alpha=0.7)  # Median reference
    ax.axhline(90, color='gray', linestyle=':', alpha=0.7)  # 90th percentile reference

    ax.set_xlabel('Number of Shots')
    ax.set_ylabel('Cumulative Win Percentage (%)')
    ax.set_title('Cumulative Distribution of Games Won')
    ax.legend()
    ax.set_xlim(17, 100)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig('results/charts/cumulative_wins.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/charts/cumulative_wins.pdf', bbox_inches='tight')
    plt.close()
    print("Created: cumulative_wins.png/pdf")


def create_violin_plot():
    """
    Create violin plot showing distribution shape.

    Combines box plot elements with kernel density estimation
    to show the full distribution shape. Wider sections indicate
    more frequent values.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    data = [prob_shots, rl_shots]
    labels = ['Probability Agent', 'RL Agent (Q-Learning)']
    colors = ['#2ecc71', '#3498db']

    # Create violin plot with mean and median markers
    parts = ax.violinplot(data, positions=[1, 2], showmeans=True, showmedians=True)

    # Color the violin bodies
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel('Shots to Win')
    ax.set_title('Distribution of Shots to Win (Violin Plot)')

    plt.tight_layout()
    plt.savefig('results/charts/violin_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/charts/violin_plot.pdf', bbox_inches='tight')
    plt.close()
    print("Created: violin_plot.png/pdf")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    print("Generating charts for paper...")
    print("-" * 40)

    # Generate all charts
    create_bar_chart()
    create_box_plot()
    create_histogram()
    create_accuracy_comparison()
    create_cumulative_wins()
    create_violin_plot()

    print("-" * 40)
    print("All charts saved to results/charts/")
    print("Both PNG (for preview) and PDF (for paper) formats created.")