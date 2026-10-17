"""
Execution Script for Phase 5: Explainability & Attention Signature Analysis.
Processes real utterances from ASVspoof 2019 using checkpoints/best_model.pt,
computes per-attack attention signatures, quantifies attention head specialization,
and caches attention matrices for visualization in notebook 05.
"""

import json
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

from src.config import config
from src.data.dataloader import create_dataloaders
from src.explainability.attention_extractor import AttentionExtractor
from src.explainability.signature_analysis import AttackSignatureAnalyzer
from src.explainability.head_specialization import HeadSpecializationAnalyzer


def main():
    print("==========================================================================")
    print(" Phase 5: Multi-Granularity Attention Explainability Pipeline             ")
    print("==========================================================================")
    
    # 1. Load trained model & extractor
    ckpt_path = config.paths.checkpoint_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Ensure Phase 4 trained model is available.")
        
    print(f"\n[1/4] Loading trained model from {ckpt_path}...")
    extractor = AttentionExtractor(checkpoint_path=ckpt_path, cfg=config)
    print("  Model loaded on device:", extractor.device)
    
    # 2. Load Evaluation DataLoader
    print("\n[2/4] Loading Evaluation DataLoader...")
    _, _, eval_loader, _ = create_dataloaders(
        cfg=config,
        use_dummy_if_missing=False,
        use_weighted_sampler=False
    )
    print(f"  Total evaluation trials: {len(eval_loader.dataset):,}")
    
    # 3. Analyze attack signatures and head specialization
    print("\n[3/4] Extracting attention maps and computing per-attack signatures...")
    sig_analyzer = AttackSignatureAnalyzer()
    head_analyzer = HeadSpecializationAnalyzer(num_fine_heads=4, num_coarse_heads=2)
    
    # Collect representative samples for each attack type (up to 50 samples per attack)
    samples_per_attack = 50
    counts_per_attack = {}
    
    cached_samples = {
        "bonafide": None,
        "A07": None,
        "A11": None,
        "A14": None,
        "A17": None,
        "A19": None
    }
    
    with torch.no_grad():
        for waveforms, labels, attacks, utts in tqdm(eval_loader, desc="Processing Eval Trials"):
            # Check if all categories reached quota
            needed = [att for att in counts_per_attack if counts_per_attack[att] < samples_per_attack]
            # If we've processed enough from each class (at least 50 each), we can break early
            if len(counts_per_attack) >= 14 and all(c >= samples_per_attack for c in counts_per_attack.values()):
                break
                
            res = extractor.extract_from_waveform(waveforms)
            
            fine_w = res["fine_frame_weights"]     # [B, 100]
            coarse_w = res["coarse_segment_weights"] # [B, 8]
            fine_heads = res["fine_heads_matrix"]    # [B, 4, 100, 100]
            coarse_heads = res["coarse_heads_matrix"] # [B, 2, 8, 8]
            
            for b in range(len(waveforms)):
                att = attacks[b]
                lbl = labels[b].item()
                category = "bonafide" if lbl == 1 else att
                
                if category not in counts_per_attack:
                    counts_per_attack[category] = 0
                    
                if counts_per_attack[category] < samples_per_attack:
                    sig_analyzer.add_sample(category, fine_w[b])
                    head_analyzer.add_sample(fine_heads[b], coarse_heads[b])
                    counts_per_attack[category] += 1
                    
                    # Cache specific high-interest examples for visualization
                    if category in cached_samples and cached_samples[category] is None:
                        cached_samples[category] = {
                            "utt_id": utts[b],
                            "prob_bonafide": float(res["probs"][b, 1]),
                            "fine_weights": fine_w[b],
                            "fine_heads": fine_heads[b],
                            "coarse_weights": coarse_w[b],
                            "coarse_heads": coarse_heads[b],
                            "lfcc": res["lfcc"][b]
                        }
                        
    # 4. Save results
    print("\n[4/4] Computing signature summaries and saving artifacts...")
    signatures = sig_analyzer.compute_all_signatures()
    head_summary = head_analyzer.get_summary()
    
    results_dir = Path(config.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    sig_path = results_dir / "explainability_signatures.json"
    with open(sig_path, "w") as f:
        json.dump(signatures, f, indent=2)
    print(f"  Saved per-attack attention signatures to {sig_path}")
    
    head_path = results_dir / "head_specialization.json"
    with open(head_path, "w") as f:
        json.dump(head_summary, f, indent=2)
    print(f"  Saved head specialization analysis to {head_path}")
    
    # Save cached sample maps for notebook
    cache_path = results_dir / "sample_attention_cache.npz"
    save_dict = {}
    for cat, data in cached_samples.items():
        if data is not None:
            save_dict[f"{cat}_fine_weights"] = data["fine_weights"]
            save_dict[f"{cat}_fine_heads"] = data["fine_heads"]
            save_dict[f"{cat}_coarse_weights"] = data["coarse_weights"]
            save_dict[f"{cat}_coarse_heads"] = data["coarse_heads"]
            save_dict[f"{cat}_lfcc"] = data["lfcc"]
            save_dict[f"{cat}_meta"] = np.array([data["utt_id"], str(data["prob_bonafide"])])
            
    np.savez_compressed(cache_path, **save_dict)
    print(f"  Saved representative attention matrices to {cache_path}")
    
    print("\n==========================================================================")
    print("  Per-Attack Attention Signature Summary:")
    print("--------------------------------------------------------------------------")
    print(f"  {'Category':12s} | {'Samples':7s} | {'Entropy':7s} | {'Top-10%':7s} | {'Onset':6s} | {'Nucleus':7s} | {'Coda':6s}")
    print("  " + "-" * 68)
    for cat, sig in signatures.items():
        print(
            f"  {cat:12s} | {sig['num_samples']:7d} | {sig['entropy']:7.4f} | "
            f"{sig['peak_top10_mass']*100:6.1f}% | {sig['onset_mass']*100:5.1f}% | "
            f"{sig['nucleus_mass']*100:6.1f}% | {sig['coda_mass']*100:5.1f}%"
        )
    print("==========================================================================")
    print("\nExplainability analysis complete!")


if __name__ == "__main__":
    main()
