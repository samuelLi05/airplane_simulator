import sys
import os
import json
import csv
from datetime import datetime

# Add db-update to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from json_update import (
    classify_prompt, 
    optimize_prompt_with_textgrad,
    _load_contrastive_classifier,
    _classify_with_centroids,
    CONTRASTIVE_DATA_PATH
)

# Path to the test bank (separate from training data)
TEST_BANK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_prompts.json")


def load_test_prompts(n_samples=40):
    """
    Load test prompts from the dedicated test bank (test_prompts.json).
    These prompts are distinct from the training data used by TextGrad.
    """
    with open(TEST_BANK_PATH, 'r') as f:
        data = json.load(f)
    
    prompts = []
    for item in data['prompts']:
        prompts.append({
            'prompt': item['prompt'],
            'gold_label': item['gold_label'],
            'category': item.get('category', 'unknown')
        })
    
    # Limit to n_samples if specified
    if n_samples and n_samples < len(prompts):
        # Take balanced samples from each class
        class_a = [p for p in prompts if p['gold_label'] == 'A']
        class_b = [p for p in prompts if p['gold_label'] == 'B']
        class_c = [p for p in prompts if p['gold_label'] == 'C']
        
        samples_per_class = n_samples // 3
        selected = class_a[:samples_per_class] + class_b[:samples_per_class] + class_c[:samples_per_class]
        
        # Fill remaining
        remaining = n_samples - len(selected)
        all_remaining = class_a[samples_per_class:] + class_b[samples_per_class:] + class_c[samples_per_class:]
        selected.extend(all_remaining[:remaining])
        
        return selected[:n_samples]
    
    return prompts


def get_similarity_details(text, model, centroid_A, centroid_B, sim_threshold=0.6):
    """Get detailed similarity info for a text."""
    result = _classify_with_centroids(text, model, centroid_A, centroid_B, sim_threshold)
    return {
        'pred_label': result['prediction'],
        'sim_A': round(result['sim_A'], 4),
        'sim_B': round(result['sim_B'], 4),
        'margin': round(result['margin'], 4),
        'best_sim': round(max(result['sim_A'], result['sim_B']), 4)
    }


def evaluate_pipeline(prompts, sim_threshold=0.6):
    """
    Evaluate the full TextGrad optimization pipeline.
    
    For each prompt:
    1. Get BEFORE similarities (original prompt)
    2. Run through optimize_prompt_with_textgrad
    3. Get AFTER similarities (optimized prompt)
    4. Compare improvement
    """
    results = []
    
    # Pre-load the contrastive model
    print("Loading contrastive classifier model...")
    model, centroid_A, centroid_B = _load_contrastive_classifier()
    
    print("\n" + "=" * 80)
    print("Evaluating TextGrad Prompt Optimization Pipeline")
    print("=" * 80)
    
    for i, item in enumerate(prompts):
        original_prompt = item['prompt']
        gold_label = item['gold_label']
        
        print(f"\n[{i+1:2d}/{len(prompts)}] Processing: {original_prompt[:60]}...")
        
        # BEFORE: Get similarity for original prompt
        before = get_similarity_details(original_prompt, model, centroid_A, centroid_B, sim_threshold)
        print(f"  BEFORE: pred={before['pred_label']} sim_A={before['sim_A']:.3f} sim_B={before['sim_B']:.3f}")
        
        # OPTIMIZE: Run through TextGrad
        try:
            optimized_prompt = optimize_prompt_with_textgrad(original_prompt)
        except Exception as e:
            print(f"  ERROR: Optimization failed: {e}")
            optimized_prompt = original_prompt  # fallback
        
        # AFTER: Get similarity for optimized prompt
        after = get_similarity_details(optimized_prompt, model, centroid_A, centroid_B, sim_threshold)
        print(f"  AFTER:  pred={after['pred_label']} sim_A={after['sim_A']:.3f} sim_B={after['sim_B']:.3f}")
        
        # Calculate improvements
        if gold_label == 'A':
            target_sim_before = before['sim_A']
            target_sim_after = after['sim_A']
        elif gold_label == 'B':
            target_sim_before = before['sim_B']
            target_sim_after = after['sim_B']
        else:  # C - use best sim
            target_sim_before = before['best_sim']
            target_sim_after = after['best_sim']
        
        sim_improvement = target_sim_after - target_sim_before
        margin_improvement = after['margin'] - before['margin']
        
        # Check if prediction improved
        before_correct = (before['pred_label'] == gold_label)
        after_correct = (after['pred_label'] == gold_label)
        
        status = ""
        if not before_correct and after_correct:
            status = "FIXED ✓"
        elif before_correct and not after_correct:
            status = "BROKE ✗"
        elif before_correct and after_correct:
            status = "OK ✓"
        else:
            status = "STILL WRONG"
        
        print(f"  DELTA:  sim_improvement={sim_improvement:+.3f} margin_improvement={margin_improvement:+.3f} [{status}]")
        print(f"  OPTIMIZED: {optimized_prompt[:80]}...")
        
        results.append({
            'index': i + 1,
            'original_prompt': original_prompt,
            'optimized_prompt': optimized_prompt,
            'gold_label': gold_label,
            'before': {
                'pred_label': before['pred_label'],
                'sim_A': before['sim_A'],
                'sim_B': before['sim_B'],
                'margin': before['margin'],
                'best_sim': before['best_sim'],
                'is_correct': before_correct
            },
            'after': {
                'pred_label': after['pred_label'],
                'sim_A': after['sim_A'],
                'sim_B': after['sim_B'],
                'margin': after['margin'],
                'best_sim': after['best_sim'],
                'is_correct': after_correct
            },
            'improvements': {
                'target_sim_delta': round(sim_improvement, 4),
                'margin_delta': round(margin_improvement, 4),
                'prediction_status': status
            }
        })
    
    return results


def compute_metrics(results):
    """Compute aggregate metrics comparing before vs after."""
    total = len(results)
    
    # Accuracy before/after
    before_correct = sum(1 for r in results if r['before']['is_correct'])
    after_correct = sum(1 for r in results if r['after']['is_correct'])
    
    # Average similarities before/after
    avg_sim_A_before = sum(r['before']['sim_A'] for r in results) / total
    avg_sim_A_after = sum(r['after']['sim_A'] for r in results) / total
    avg_sim_B_before = sum(r['before']['sim_B'] for r in results) / total
    avg_sim_B_after = sum(r['after']['sim_B'] for r in results) / total
    avg_margin_before = sum(r['before']['margin'] for r in results) / total
    avg_margin_after = sum(r['after']['margin'] for r in results) / total
    
    # Average target similarity improvement
    avg_sim_improvement = sum(r['improvements']['target_sim_delta'] for r in results) / total
    avg_margin_improvement = sum(r['improvements']['margin_delta'] for r in results) / total
    
    # Count improvements vs regressions
    improved = sum(1 for r in results if r['improvements']['target_sim_delta'] > 0)
    regressed = sum(1 for r in results if r['improvements']['target_sim_delta'] < 0)
    unchanged = total - improved - regressed
    
    # Prediction changes
    fixed = sum(1 for r in results if r['improvements']['prediction_status'] == "FIXED ✓")
    broke = sum(1 for r in results if r['improvements']['prediction_status'] == "BROKE ✗")
    
    # Per-class metrics
    class_metrics = {}
    for label in ['A', 'B', 'C']:
        class_results = [r for r in results if r['gold_label'] == label]
        if class_results:
            class_metrics[label] = {
                'total': len(class_results),
                'before_accuracy': sum(1 for r in class_results if r['before']['is_correct']) / len(class_results),
                'after_accuracy': sum(1 for r in class_results if r['after']['is_correct']) / len(class_results),
                'avg_sim_improvement': sum(r['improvements']['target_sim_delta'] for r in class_results) / len(class_results),
                'avg_margin_improvement': sum(r['improvements']['margin_delta'] for r in class_results) / len(class_results)
            }
    
    return {
        'total_samples': total,
        'accuracy': {
            'before': round(before_correct / total, 4),
            'after': round(after_correct / total, 4),
            'before_correct': before_correct,
            'after_correct': after_correct
        },
        'avg_similarities': {
            'sim_A_before': round(avg_sim_A_before, 4),
            'sim_A_after': round(avg_sim_A_after, 4),
            'sim_B_before': round(avg_sim_B_before, 4),
            'sim_B_after': round(avg_sim_B_after, 4),
            'margin_before': round(avg_margin_before, 4),
            'margin_after': round(avg_margin_after, 4)
        },
        'improvements': {
            'avg_target_sim_delta': round(avg_sim_improvement, 4),
            'avg_margin_delta': round(avg_margin_improvement, 4),
            'samples_improved': improved,
            'samples_regressed': regressed,
            'samples_unchanged': unchanged,
            'predictions_fixed': fixed,
            'predictions_broke': broke
        },
        'per_class': class_metrics
    }


def save_results(results, metrics, output_path):
    """Save results to JSON file."""
    output = {
        'timestamp': datetime.now().isoformat(),
        'evaluation_type': 'textgrad_pipeline_before_after',
        'config': {
            'n_samples': len(results),
            'sim_threshold': 0.6,
            'contrastive_model': 'notebooks/air_sim_model_v3',
            'optimizer': 'TextGrad with DPO foundation',
            'optimizer_max_tokens': 150,
            'optimizer_temperature': 0.3,
            'test_bank': 'db-update/test_prompts.json',
            'note': 'Test prompts use simulator vocabulary (routes, airports, route_available, cost, time_estimate, origin_airport_id, destination_airport_id, route_id)'
        },
        'summary': {
            'accuracy_before': f"{metrics['accuracy']['before']*100:.1f}%",
            'accuracy_after': f"{metrics['accuracy']['after']*100:.1f}%",
            'accuracy_improvement': f"{(metrics['accuracy']['after'] - metrics['accuracy']['before'])*100:+.1f}%",
            'avg_similarity_improvement': f"{metrics['improvements']['avg_target_sim_delta']:+.4f}",
            'avg_margin_improvement': f"{metrics['improvements']['avg_margin_delta']:+.4f}",
            'samples_improved': f"{metrics['improvements']['samples_improved']}/{metrics['total_samples']}",
            'predictions_fixed': metrics['improvements']['predictions_fixed'],
            'predictions_broke': metrics['improvements']['predictions_broke']
        },
        'detailed_metrics': metrics,
        'detailed_results': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def print_summary(metrics):
    """Print evaluation summary."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY: TextGrad Prompt Optimization Pipeline")
    print("=" * 80)
    
    acc = metrics['accuracy']
    print(f"\nACCURACY:")
    print(f"  Before optimization: {acc['before']*100:.1f}% ({acc['before_correct']}/{metrics['total_samples']})")
    print(f"  After optimization:  {acc['after']*100:.1f}% ({acc['after_correct']}/{metrics['total_samples']})")
    print(f"  Improvement:         {(acc['after'] - acc['before'])*100:+.1f}%")
    
    sims = metrics['avg_similarities']
    print(f"\nAVERAGE SIMILARITIES:")
    print(f"  sim_A: {sims['sim_A_before']:.4f} → {sims['sim_A_after']:.4f} ({sims['sim_A_after'] - sims['sim_A_before']:+.4f})")
    print(f"  sim_B: {sims['sim_B_before']:.4f} → {sims['sim_B_after']:.4f} ({sims['sim_B_after'] - sims['sim_B_before']:+.4f})")
    print(f"  margin: {sims['margin_before']:.4f} → {sims['margin_after']:.4f} ({sims['margin_after'] - sims['margin_before']:+.4f})")
    
    impr = metrics['improvements']
    print(f"\nIMPROVEMENTS:")
    print(f"  Avg target similarity delta: {impr['avg_target_sim_delta']:+.4f}")
    print(f"  Avg margin delta:            {impr['avg_margin_delta']:+.4f}")
    print(f"  Samples improved:  {impr['samples_improved']}/{metrics['total_samples']}")
    print(f"  Samples regressed: {impr['samples_regressed']}/{metrics['total_samples']}")
    print(f"  Predictions fixed: {impr['predictions_fixed']}")
    print(f"  Predictions broke: {impr['predictions_broke']}")
    
    print(f"\nPER-CLASS RESULTS:")
    for label in ['A', 'B', 'C']:
        if label in metrics['per_class']:
            cls = metrics['per_class'][label]
            print(f"  Class {label} (n={cls['total']}):")
            print(f"    Accuracy: {cls['before_accuracy']*100:.1f}% → {cls['after_accuracy']*100:.1f}%")
            print(f"    Avg sim improvement: {cls['avg_sim_improvement']:+.4f}")
            print(f"    Avg margin improvement: {cls['avg_margin_improvement']:+.4f}")
    
    print("=" * 80)


def main():
    print("=" * 80)
    print("TextGrad Prompt Optimization Pipeline Evaluation")
    print("Comparing BEFORE vs AFTER similarity scores")
    print("=" * 80)
    print(f"\nUsing TEST BANK: {TEST_BANK_PATH}")
    print("(These prompts are distinct from training data)")
    
    # Load test prompts from dedicated test bank
    print("\nLoading test prompts...")
    prompts = load_test_prompts(n_samples=40)  # Use all 40 test prompts
    print(f"Loaded {len(prompts)} prompts for evaluation")
    
    # Count per class
    class_counts = {}
    for label in ['A', 'B', 'C']:
        class_counts[label] = sum(1 for p in prompts if p['gold_label'] == label)
    print(f"Class distribution: A={class_counts['A']}, B={class_counts['B']}, C={class_counts['C']}")
    
    # Evaluate pipeline
    results = evaluate_pipeline(prompts)
    
    # Compute metrics
    metrics = compute_metrics(results)
    
    # Print summary
    print_summary(metrics)
    
    # Save results
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')
    save_results(results, metrics, output_path)
    
    return results, metrics


if __name__ == "__main__":
    main()
