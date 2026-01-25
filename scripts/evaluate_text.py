#!/usr/bin/env python3
"""
Evaluation script for ReMDM experiment results.

Purpose: Convert raw experiment outputs into thesis-ready metrics table.

Usage:
    python scripts/evaluate_text.py results/20260125_145111_remdm/
    python scripts/evaluate_text.py results/  # Evaluate all runs
    python scripts/evaluate_text.py results/ --output metrics.csv
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import Counter
import csv
import sys

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def compute_distinct_n(tokens: List[str], n: int) -> float:
    """
    Compute distinct-n: unique n-grams / total n-grams.
    Higher = more diverse vocabulary.
    """
    if len(tokens) < n:
        return 0.0
    
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    
    return len(set(ngrams)) / len(ngrams)


def compute_repetition_metrics(text: str) -> Dict[str, float]:
    """Compute simple repetition/diversity metrics."""
    tokens = text.split()
    
    if not tokens:
        return {
            'distinct_1': 0.0,
            'distinct_2': 0.0,
            'num_tokens': 0,
            'num_chars': 0,
        }
    
    return {
        'distinct_1': compute_distinct_n(tokens, 1),
        'distinct_2': compute_distinct_n(tokens, 2),
        'num_tokens': len(tokens),
        'num_chars': len(text),
    }


def evaluate_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Evaluate a single experiment run.
    
    Returns metrics dict or None if run incomplete/invalid.
    """
    run_dir = Path(run_dir)
    
    # Check if run is complete
    summary_path = run_dir / "summary.json"
    gen_seqs_path = run_dir / "external_remdm" / "generated_sequences.json"
    
    if not summary_path.exists():
        logger.warning(f"No summary.json in {run_dir.name}, skipping")
        return None
    
    if not gen_seqs_path.exists():
        logger.warning(f"No generated_sequences.json in {run_dir.name}, skipping")
        return None
    
    # Load metadata
    with open(summary_path) as f:
        summary = json.load(f)
    
    # Load generated text
    with open(gen_seqs_path) as f:
        gen_data = json.load(f)
    
    # Extract basic info
    strategy = summary.get('strategy', 'unknown')
    steps = summary.get('steps', 0)
    
    # Get upstream metrics if available
    gen_ppl = gen_data.get('gen_ppl', None)
    mauve = gen_data.get('MAUVE', None)
    entropy = gen_data.get('entropy', None)
    
    # Get text samples
    text_samples = gen_data.get('text_samples', [])
    num_samples = len(text_samples)
    
    if num_samples == 0:
        logger.warning(f"No text samples in {run_dir.name}")
        return None
    
    # Compute text metrics across all samples
    all_tokens = []
    all_chars = []
    sample_lengths = []
    
    for sample in text_samples:
        # Remove <|endoftext|> markers for cleaner analysis
        clean_sample = sample.replace('<|endoftext|>', ' ')
        tokens = clean_sample.split()
        
        all_tokens.extend(tokens)
        all_chars.append(len(clean_sample))
        sample_lengths.append(len(tokens))
    
    # Compute diversity metrics on concatenated text
    if all_tokens:
        distinct_1 = compute_distinct_n(all_tokens, 1)
        distinct_2 = compute_distinct_n(all_tokens, 2)
    else:
        distinct_1 = 0.0
        distinct_2 = 0.0
    
    avg_length_tokens = sum(sample_lengths) / len(sample_lengths) if sample_lengths else 0
    avg_length_chars = sum(all_chars) / len(all_chars) if all_chars else 0
    
    # Compile metrics
    metrics = {
        'run_name': run_dir.name,
        'strategy': strategy,
        'steps': steps,
        'num_samples': num_samples,
        'avg_length_tokens': round(avg_length_tokens, 1),
        'avg_length_chars': round(avg_length_chars, 1),
        'distinct_1': round(distinct_1, 4),
        'distinct_2': round(distinct_2, 4),
        'gen_ppl': round(gen_ppl, 2) if gen_ppl is not None else None,
        'mauve': round(mauve, 4) if mauve is not None else None,
        'entropy': round(entropy, 2) if entropy is not None else None,
    }
    
    return metrics


def evaluate_all_runs(results_dir: Path) -> List[Dict[str, Any]]:
    """Evaluate all runs in results directory."""
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return []
    
    # Find all run directories (format: YYYYMMDD_HHMMSS_remdm)
    run_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and '_remdm' in d.name])
    
    logger.info(f"Found {len(run_dirs)} run directories")
    
    all_metrics = []
    for run_dir in run_dirs:
        metrics = evaluate_run(run_dir)
        if metrics:
            all_metrics.append(metrics)
            logger.info(f"✓ Evaluated {run_dir.name}: {metrics['strategy']}, {metrics['num_samples']} samples")
    
    return all_metrics


def print_metrics_table(metrics_list: List[Dict[str, Any]]) -> None:
    """Print metrics in a readable table format."""
    if not metrics_list:
        print("No metrics to display")
        return
    
    # Group by strategy for easier comparison
    by_strategy = {}
    for m in metrics_list:
        strategy = m['strategy']
        if strategy not in by_strategy:
            by_strategy[strategy] = []
        by_strategy[strategy].append(m)
    
    print("\n" + "=" * 100)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 100)
    
    for strategy, runs in sorted(by_strategy.items()):
        print(f"\n{strategy.upper()}")
        print("-" * 100)
        
        for m in runs:
            print(f"  Run: {m['run_name']}")
            print(f"    Steps: {m['steps']}, Samples: {m['num_samples']}")
            print(f"    Avg Length: {m['avg_length_tokens']} tokens ({m['avg_length_chars']} chars)")
            print(f"    Diversity: distinct-1={m['distinct_1']:.4f}, distinct-2={m['distinct_2']:.4f}")
            
            if m['gen_ppl'] is not None:
                print(f"    Perplexity: {m['gen_ppl']:.2f}")
            if m['mauve'] is not None:
                print(f"    MAUVE: {m['mauve']:.4f}")
            if m['entropy'] is not None:
                print(f"    Entropy: {m['entropy']:.2f}")
            print()
    
    # Summary comparison table
    print("\n" + "=" * 100)
    print("STRATEGY COMPARISON")
    print("=" * 100)
    print(f"{'Strategy':<20} {'Steps':<8} {'Samples':<10} {'Distinct-1':<12} {'Distinct-2':<12} {'PPL':<10} {'MAUVE':<10}")
    print("-" * 100)
    
    for m in sorted(metrics_list, key=lambda x: (x['strategy'], x['steps'])):
        ppl_str = f"{m['gen_ppl']:.2f}" if m['gen_ppl'] is not None else "N/A"
        mauve_str = f"{m['mauve']:.4f}" if m['mauve'] is not None else "N/A"
        
        print(f"{m['strategy']:<20} {m['steps']:<8} {m['num_samples']:<10} "
              f"{m['distinct_1']:<12.4f} {m['distinct_2']:<12.4f} "
              f"{ppl_str:<10} {mauve_str:<10}")
    
    print("=" * 100 + "\n")


def save_metrics_csv(metrics_list: List[Dict[str, Any]], output_path: Path) -> None:
    """Save metrics to CSV file."""
    if not metrics_list:
        logger.warning("No metrics to save")
        return
    
    fieldnames = [
        'run_name', 'strategy', 'steps', 'num_samples',
        'avg_length_tokens', 'avg_length_chars',
        'distinct_1', 'distinct_2',
        'gen_ppl', 'mauve', 'entropy'
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_list)
    
    logger.info(f"Saved metrics to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate ReMDM experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single run
  python scripts/evaluate_text.py results/20260125_145111_remdm/
  
  # Evaluate all runs
  python scripts/evaluate_text.py results/
  
  # Save to CSV
  python scripts/evaluate_text.py results/ --output metrics.csv
  
  # Quiet mode (only CSV output)
  python scripts/evaluate_text.py results/ --output metrics.csv --quiet
        """
    )
    
    parser.add_argument(
        'path',
        type=Path,
        help='Path to results directory or single run directory'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Save metrics to CSV file'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress table output (useful with --output)'
    )
    
    args = parser.parse_args()
    
    # Determine if single run or directory
    if args.path.is_dir() and (args.path / 'summary.json').exists():
        # Single run
        metrics = evaluate_run(args.path)
        metrics_list = [metrics] if metrics else []
    else:
        # Directory of runs
        metrics_list = evaluate_all_runs(args.path)
    
    if not metrics_list:
        logger.error("No valid runs found")
        sys.exit(1)
    
    # Print table
    if not args.quiet:
        print_metrics_table(metrics_list)
    
    # Save CSV
    if args.output:
        save_metrics_csv(metrics_list, args.output)
    
    logger.info(f"Evaluated {len(metrics_list)} runs successfully")


if __name__ == '__main__':
    main()
