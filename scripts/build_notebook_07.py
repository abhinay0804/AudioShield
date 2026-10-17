"""
Builds and executes notebooks/07_demo.ipynb.
Provides an interactive end-to-end inference and explainability demo:
1. Loads best model checkpoint (checkpoints/best_model.pt)
2. Loads genuine and spoof audio from ASVspoof 2019 LA dataset
3. Predicts genuine vs spoof probability with calibrated threshold
4. Visualizes waveform, LFCC spectrogram, fine frame attention, and coarse segment attention
"""

import nbformat as nbf
from pathlib import Path


def build_demo_notebook():
    nb = nbf.v4.new_notebook()
    cells = []
    
    # Cell 1: Markdown Title
    cells.append(nbf.v4.new_markdown_cell(
        "# Phase 6: Interactive Audio Deepfake Detection Demo\n"
        "### Dual-Stream Temporal Attention & Explainability Inference Pipeline\n\n"
        "This interactive demo loads our trained **Dual-Stream CNN-BiLSTM + Multi-Head Self-Attention** model "
        "(`checkpoints/best_model.pt`) and performs end-to-end detection on arbitrary audio samples from the official "
        "ASVspoof 2019 LA evaluation dataset.\n\n"
        "### Output Deliverables:\n"
        "1. **Binary Classification Verdict**: Genuine (Bonafide) vs. Deepfake (Spoof)\n"
        "2. **Confidence Score & Calibrated Operating Margin**\n"
        "3. **Dual-Granularity Explainability Display**: Temporal frame saliency + Segment attention distribution"
    ))
    
    # Cell 2: Imports & Model Setup
    cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "sys.path.insert(0, '..')\n"
        "import json\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import torch\n"
        "import soundfile as sf\n"
        "import torchaudio\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "\n"
        "from src.config import config\n"
        "from src.models.dual_stream_model import DualStreamModel\n"
        "from src.explainability.attention_extractor import AttentionExtractor\n"
        "\n"
        "sns.set_theme(style='white', font_scale=1.1)\n"
        "plt.rcParams['figure.dpi'] = 120\n"
        "\n"
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
        "print(f'Running on device: {device}')\n"
        "\n"
        "# Load best checkpoint and AttentionExtractor\n"
        "ckpt_path = Path('../checkpoints/best_model.pt')\n"
        "assert ckpt_path.exists(), f'Checkpoint not found at {ckpt_path}'\n"
        "\n"
        "model = DualStreamModel(config.model).to(device)\n"
        "ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)\n"
        "model.load_state_dict(ckpt['model_state_dict'])\n"
        "model.eval()\n"
        "\n"
        "attn_extractor = AttentionExtractor(model=model, cfg=config)\n"
        "\n"
        "# Load optimal threshold\n"
        "metrics_path = Path('../results/final_evaluation_metrics.json')\n"
        "with open(metrics_path) as f:\n"
        "    eval_metrics = json.load(f)\n"
        "threshold = eval_metrics['threshold']\n"
        "print(f'Model initialized successfully! Operating decision threshold: {threshold:.4f}')"
    ))
    
    # Cell 3: Prediction & Visualization Function
    cells.append(nbf.v4.new_code_cell(
        "def analyze_audio_trial(audio_path, ground_truth=None, attack_type=None):\n"
        "    \"\"\"\n"
        "    Executes inference and dual-granularity explainability for an audio trial.\n"
        "    \"\"\"\n"
        "    audio_path = Path(audio_path)\n"
        "    assert audio_path.exists(), f'Audio file {audio_path} does not exist!'\n"
        "    \n"
        "    # 1. Load and preprocess audio using soundfile\n"
        "    data, sr = sf.read(str(audio_path))\n"
        "    waveform = torch.from_numpy(data).float()\n"
        "    if waveform.ndim == 1:\n"
        "        waveform = waveform.unsqueeze(0)\n"
        "    else:\n"
        "        waveform = waveform.t()\n"
        "    if waveform.shape[0] > 1:\n"
        "        waveform = torch.mean(waveform, dim=0, keepdim=True)\n"
        "    if sr != config.audio.sample_rate:\n"
        "        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=config.audio.sample_rate)\n"
        "        \n"
        "    target_len = config.audio.target_samples\n"
        "    if waveform.shape[1] < target_len:\n"
        "        pad = target_len - waveform.shape[1]\n"
        "        waveform = torch.nn.functional.pad(waveform, (0, pad))\n"
        "    else:\n"
        "        waveform = waveform[:, :target_len]\n"
        "        \n"
        "    # 2. Extract Explainability & Forward Pass\n"
        "    res = attn_extractor.extract_from_waveform(waveform)\n"
        "    bonafide_prob = float(res['probs'][0, 1])\n"
        "    is_bonafide_pred = (bonafide_prob >= threshold)\n"
        "    pred_label = 'GENUINE (Bonafide)' if is_bonafide_pred else 'DEEPFAKE (Spoof)'\n"
        "    \n"
        "    # 3. Print Results\n"
        "    print('========================================================================')\n"
        "    print(f' Trial: {audio_path.name} | Ground Truth: {ground_truth} | Attack: {attack_type}')\n"
        "    print('========================================================================')\n"
        "    print(f'  • Verdict:           {pred_label}')\n"
        "    print(f'  • Bonafide Prob:     {bonafide_prob*100:6.2f}%')\n"
        "    print(f'  • Spoof Prob:        {(1.0 - bonafide_prob)*100:6.2f}%')\n"
        "    print(f'  • Decision Boundary: {threshold:6.4f}')\n"
        "    print('------------------------------------------------------------------------')\n"
        "    \n"
        "    # 4. Multi-Granularity Saliency Plots\n"
        "    fig = plt.figure(figsize=(15, 10))\n"
        "    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.3, 1.2], width_ratios=[1, 1])\n"
        "    \n"
        "    # Subplot 1: Waveform\n"
        "    ax_wave = fig.add_subplot(gs[0, :])\n"
        "    time_axis = np.linspace(0, 4.0, target_len)\n"
        "    ax_wave.plot(time_axis, waveform.squeeze().cpu().numpy(), color='#1f77b4', lw=0.8)\n"
        "    ax_wave.set_xlim([0, 4.0])\n"
        "    ax_wave.set_title(f'Input Audio Waveform (4.0s @ 16kHz) — Prediction: {pred_label}', fontweight='bold')\n"
        "    ax_wave.set_xlabel('Time [seconds]')\n"
        "    ax_wave.set_ylabel('Amplitude')\n"
        "    ax_wave.grid(True, alpha=0.3)\n"
        "    \n"
        "    # Subplot 2: LFCC Spectrogram with Frame-level Attention Saliency Overlay\n"
        "    ax_spec = fig.add_subplot(gs[1, :])\n"
        "    lfcc_mat = res['lfcc'][0]  # [60, 401]\n"
        "    im = ax_spec.imshow(lfcc_mat, aspect='auto', origin='lower', extent=[0, 4.0, 0, 60], cmap='viridis')\n"
        "    \n"
        "    # Overlay Saliency curve\n"
        "    ax_saliency = ax_spec.twinx()\n"
        "    fine_w = res['fine_frame_weights'][0]  # [100]\n"
        "    t_orig = np.linspace(0, 4.0, len(fine_w))\n"
        "    t_interp = np.linspace(0, 4.0, lfcc_mat.shape[1])\n"
        "    saliency = np.interp(t_interp, t_orig, fine_w)\n"
        "    ax_saliency.plot(t_interp, saliency, color='red', lw=2.5, label='Attention Saliency Profile')\n"
        "    ax_saliency.set_ylabel('Attention Saliency', color='red', fontweight='bold')\n"
        "    ax_saliency.tick_params(axis='y', labelcolor='red')\n"
        "    ax_saliency.set_ylim([0, max(saliency) * 1.5])\n"
        "    ax_saliency.legend(loc='upper right')\n"
        "    \n"
        "    ax_spec.set_title('LFCC Spectrogram with Frame-Level Temporal Attention Saliency (Fine Stream ~10ms)', fontweight='bold')\n"
        "    ax_spec.set_xlabel('Time [seconds]')\n"
        "    ax_spec.set_ylabel('LFCC Filterbank (Static+Δ+ΔΔ)')\n"
        "    \n"
        "    # Subplot 3: Fine Head Attention Matrix\n"
        "    ax_fine = fig.add_subplot(gs[2, 0])\n"
        "    fine_h1 = res['fine_heads_matrix'][0, 0]  # [100, 100]\n"
        "    im_f = ax_fine.imshow(fine_h1, cmap='magma' if not is_bonafide_pred else 'viridis', aspect='auto', origin='lower')\n"
        "    ax_fine.set_title('Fine-Grained Attention Matrix (Head 1: 100x100)', fontweight='bold')\n"
        "    ax_fine.set_xlabel('Key Frame (~40ms step)')\n"
        "    ax_fine.set_ylabel('Query Frame')\n"
        "    plt.colorbar(im_f, ax=ax_fine, shrink=0.8)\n"
        "    \n"
        "    # Subplot 4: Coarse Segment Attention Distribution\n"
        "    ax_coarse = fig.add_subplot(gs[2, 1])\n"
        "    seg_weights = res['coarse_segment_weights'][0]  # [8]\n"
        "    seg_idx = [f'Seg {i+1}\\n({i*0.5:.1f}-{(i+1)*0.5:.1f}s)' for i in range(len(seg_weights))]\n"
        "    bars = ax_coarse.bar(seg_idx, seg_weights, color='#ff7f0e' if not is_bonafide_pred else '#2ca02c', edgecolor='black')\n"
        "    ax_coarse.set_title('Coarse-Grained Segment Attention (8 Segments ~480ms each)', fontweight='bold')\n"
        "    ax_coarse.set_ylabel('Segment Attention Weight')\n"
        "    ax_coarse.set_ylim([0, max(seg_weights) * 1.3])\n"
        "    for b in bars:\n"
        "        y = b.get_height()\n"
        "        ax_coarse.text(b.get_x() + b.get_width()/2.0, y + 0.005, f'{y:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')\n"
        "    ax_coarse.grid(axis='y', alpha=0.3)\n"
        "    \n"
        "    plt.suptitle(f'Explainability Diagnostic Dashboard — Verdict: {pred_label}', fontsize=15, fontweight='bold', y=0.99)\n"
        "    plt.tight_layout()\n"
        "    plt.show()"
    ))
    
    # Cell 4: Demo Run 1 - Genuine (Bonafide) Trial
    cells.append(nbf.v4.new_code_cell(
        "# Demo Trial 1: Genuine Human Speech (Bonafide)\n"
        "eval_audio_dir = Path('../data/LA/ASVspoof2019_LA_eval/flac')\n"
        "eval_proto = Path('../data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt')\n"
        "\n"
        "# Find first genuine trial from protocol\n"
        "bonafide_utt = None\n"
        "with open(eval_proto) as f:\n"
        "    for line in f:\n"
        "        parts = line.strip().split()\n"
        "        if len(parts) >= 5 and parts[4] == 'bonafide':\n"
        "            candidate = eval_audio_dir / f'{parts[1]}.flac'\n"
        "            if candidate.exists():\n"
        "                bonafide_utt = (candidate, 'bonafide', 'None')\n"
        "                break\n"
        "\n"
        "if bonafide_utt:\n"
        "    analyze_audio_trial(bonafide_utt[0], ground_truth=bonafide_utt[1], attack_type=bonafide_utt[2])\n"
        "else:\n"
        "    print('No bonafide file found!')"
    ))
    
    # Cell 5: Demo Run 2 - Deepfake Trial (Attack A07 - WaveNet TTS)
    cells.append(nbf.v4.new_code_cell(
        "# Demo Trial 2: Deepfake Speech (Attack A07 - Neural TTS WaveNet vocoder)\n"
        "a07_utt = None\n"
        "with open(eval_proto) as f:\n"
        "    for line in f:\n"
        "        parts = line.strip().split()\n"
        "        if len(parts) >= 5 and parts[3] == 'A07':\n"
        "            candidate = eval_audio_dir / f'{parts[1]}.flac'\n"
        "            if candidate.exists():\n"
        "                a07_utt = (candidate, 'spoof', 'A07 (WaveNet)')\n"
        "                break\n"
        "\n"
        "if a07_utt:\n"
        "    analyze_audio_trial(a07_utt[0], ground_truth=a07_utt[1], attack_type=a07_utt[2])\n"
        "else:\n"
        "    print('No A07 file found!')"
    ))
    
    # Cell 6: Demo Run 3 - Deepfake Trial (Attack A11 - Griffin-Lim TTS)
    cells.append(nbf.v4.new_code_cell(
        "# Demo Trial 3: Deepfake Speech (Attack A11 - Griffin-Lim vocoder)\n"
        "a11_utt = None\n"
        "with open(eval_proto) as f:\n"
        "    for line in f:\n"
        "        parts = line.strip().split()\n"
        "        if len(parts) >= 5 and parts[3] == 'A11':\n"
        "            candidate = eval_audio_dir / f'{parts[1]}.flac'\n"
        "            if candidate.exists():\n"
        "                a11_utt = (candidate, 'spoof', 'A11 (Griffin-Lim)')\n"
        "                break\n"
        "\n"
        "if a11_utt:\n"
        "    analyze_audio_trial(a11_utt[0], ground_truth=a11_utt[1], attack_type=a11_utt[2])\n"
        "else:\n"
        "    print('No A11 file found!')"
    ))
    
    nb.cells = cells
    
    out_dir = Path("notebooks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "07_demo.ipynb"
    
    with open(out_file, "w") as f:
        nbf.write(nb, f)
        
    print(f"Successfully generated {out_file} with {len(cells)} cells.")


if __name__ == "__main__":
    build_demo_notebook()
