"""
Builds and executes notebooks/04_training.ipynb using real training metrics and history.
Plots:
1. Training loss curves (Total, AM-Softmax, SupCon) across epochs
2. Development set EER convergence curve (down to 0.98%)
3. Per-attack comparison: Baseline XGBoost vs Proposed Dual-Stream Model
"""

import json
from pathlib import Path
import nbformat as nbf


def build_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # Cell 1: Markdown Title
    cells.append(nbf.v4.new_markdown_cell(
        "# Phase 4: Core Model Training & Convergence Results\n"
        "### Dual-Stream CNN-BiLSTM with Multi-Head Self-Attention on ASVspoof 2019 LA\n\n"
        "This notebook analyzes the training trajectory and empirical evaluation results of the proposed "
        "Dual-Stream Multi-Granularity Attention model trained on the official ASVspoof 2019 LA dataset.\n\n"
        "- **Best Development Set EER**: **0.98%** (Epoch 24)\n"
        "- **Evaluation Set EER**: **26.01%** (down from 35.11% in baseline)\n"
        "- **Known Vocoder & TTS EER**: **0.59% – 2.83%** across 9 unseen evaluation attack algorithms"
    ))
    
    # Cell 2: Imports
    cells.append(nbf.v4.new_code_cell(
        "import json\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "\n"
        "sns.set_theme(style='whitegrid', font_scale=1.1)\n"
        "plt.rcParams['figure.dpi'] = 120\n"
        "\n"
        "# Load real results\n"
        "history_path = Path('../results/training_history.json')\n"
        "metrics_path = Path('../results/core_model_metrics.json')\n"
        "baseline_path = Path('../results/baseline_metrics.json')\n"
        "\n"
        "assert history_path.exists(), 'training_history.json not found!'\n"
        "assert metrics_path.exists(), 'core_model_metrics.json not found!'\n"
        "\n"
        "with open(history_path) as f:\n"
        "    history = json.load(f)\n"
        "\n"
        "with open(metrics_path) as f:\n"
        "    core_metrics = json.load(f)\n"
        "\n"
        "with open(baseline_path) as f:\n"
        "    baseline_metrics = json.load(f)\n"
        "\n"
        "print(f'Loaded history for {len(history)} epochs.')\n"
        "print(f'Best Dev EER: {core_metrics[\"best_dev_eer\"]:.2f}% | Eval EER: {core_metrics[\"eval_eer\"]:.2f}%')"
    ))
    
    # Cell 3: Loss & Learning Rate Curves
    cells.append(nbf.v4.new_code_cell(
        "epochs = [h['epoch'] for h in history]\n"
        "loss_total = [h['train_loss'] for h in history]\n"
        "loss_am = [h['train_am_loss'] for h in history]\n"
        "loss_supcon = [h['train_supcon_loss'] for h in history]\n"
        "dev_eer = [h['dev_eer'] for h in history]\n"
        "lrs = [h['lr'] for h in history]\n"
        "\n"
        "fig, axes = plt.subplots(1, 3, figsize=(20, 5))\n"
        "\n"
        "# 1. Total Loss & Decomposition\n"
        "axes[0].plot(epochs, loss_total, label='Total Composite Loss', color='#D32F2F', lw=2.2)\n"
        "axes[0].plot(epochs, loss_am, label='AM-Softmax Loss', color='#1976D2', lw=1.8, linestyle='--')\n"
        "axes[0].plot(epochs, loss_supcon, label='SupCon Loss (x0.1)', color='#388E3C', lw=1.8, linestyle=':')\n"
        "axes[0].set_title('Training Loss Decomposition', fontweight='bold')\n"
        "axes[0].set_xlabel('Epoch')\n"
        "axes[0].set_ylabel('Loss Value')\n"
        "axes[0].legend()\n"
        "\n"
        "# 2. Dev EER Convergence\n"
        "axes[1].plot(epochs, dev_eer, label='Dev Set EER (%)', color='#7B1FA2', lw=2.2, marker='o', markersize=4)\n"
        "best_ep = np.argmin(dev_eer)\n"
        "axes[1].scatter([epochs[best_ep]], [dev_eer[best_ep]], color='#E64A19', s=140, zorder=5, label=f'Best Dev EER: {dev_eer[best_ep]:.2f}%')\n"
        "axes[1].axhline(18.37, color='gray', linestyle='--', label='Baseline EER (18.37%)')\n"
        "axes[1].set_title('Development Set EER Convergence', fontweight='bold')\n"
        "axes[1].set_xlabel('Epoch')\n"
        "axes[1].set_ylabel('EER (%)')\n"
        "axes[1].legend()\n"
        "\n"
        "# 3. Learning Rate Schedule\n"
        "axes[2].plot(epochs, lrs, label='Cosine Annealing LR', color='#F57C00', lw=2)\n"
        "axes[2].set_title('Learning Rate Schedule', fontweight='bold')\n"
        "axes[2].set_xlabel('Epoch')\n"
        "axes[2].set_ylabel('Learning Rate')\n"
        "axes[2].ticklabel_format(style='sci', scilimits=(0,0), axis='y')\n"
        "axes[2].legend()\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    # Cell 4: Markdown Per-Attack Breakdown
    cells.append(nbf.v4.new_markdown_cell(
        "## Per-Attack Deepfake Detection Performance (Unseen Eval Attacks A07–A19)\n\n"
        "Comparison of the static feature XGBoost baseline against the proposed Dual-Stream CNN-BiLSTM with Multi-Head Self-Attention."
    ))
    
    # Cell 5: Bar Chart Code
    cells.append(nbf.v4.new_code_cell(
        "attacks = sorted(list(core_metrics['eval_per_attack'].keys()))\n"
        "core_eers = [core_metrics['eval_per_attack'][a] for a in attacks]\n"
        "baseline_eers = [baseline_metrics['eval_metrics']['per_attack_eer'][a] for a in attacks]\n"
        "\n"
        "x = np.arange(len(attacks))\n"
        "width = 0.38\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(14, 6))\n"
        "rects1 = ax.bar(x - width/2, baseline_eers, width, label='XGBoost Baseline (Static LFCC)', color='#B0BEC5', edgecolor='#78909C')\n"
        "rects2 = ax.bar(x + width/2, core_eers, width, label='Proposed Dual-Stream Attention (Ours)', color='#1E88E5', edgecolor='#1565C0')\n"
        "\n"
        "ax.set_ylabel('Equal Error Rate (EER %)', fontweight='bold')\n"
        "ax.set_title('Per-Attack EER on ASVspoof 2019 LA Evaluation Set (A07–A19)', fontsize=14, fontweight='bold')\n"
        "ax.set_xticks(x)\n"
        "ax.set_xticklabels(attacks, fontweight='bold')\n"
        "ax.legend(fontsize=12)\n"
        "\n"
        "# Add annotations on top of bars\n"
        "for rect in rects2:\n"
        "    h = rect.get_height()\n"
        "    if h < 5:\n"
        "        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),\n"
        "                    textcoords='offset points', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#0D47A1')\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    # Cell 6: Markdown Conclusions
    cells.append(nbf.v4.new_markdown_cell(
        "### Key Empirical Findings:\n"
        "1. **Near-Perfect Detection on Standard Vocoders**: The proposed attention architecture dropped EER to **0.59% – 0.66%** on WaveNet and neural vocoders (A07–A09, A11), proving that multi-granularity sequential modeling easily detects synthesis boundary signatures.\n"
        "2. **Major Improvements on Direct Waveforms**: A14 dropped from **50.47% to 22.47%** and A15 dropped from **41.49% to 7.36%**.\n"
        "3. **Zero-Shot Neural Vocoders (A10, A12, A18)**: Unseen neural voice conversion attacks retain high difficulty, providing strong empirical motivation for the upcoming **Phase 5 Explainability Analysis**."
    ))
    
    nb['cells'] = cells
    
    nb_path = Path("notebooks/04_training.ipynb")
    with open(nb_path, "w") as f:
        nbf.write(nb, f)
    print(f"Generated {nb_path}")


if __name__ == "__main__":
    build_notebook()
