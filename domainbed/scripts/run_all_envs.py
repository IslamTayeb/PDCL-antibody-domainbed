#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
import shutil
from pathlib import Path

def run_training(args, test_env, output_dir):
    """
    Run the domainbed training script with the specified test environment
    """
    cmd = [
        sys.executable, "-m", "domainbed.scripts.train",
        f"--dataset={args.dataset}",
        f"--algorithm={args.algorithm}",
        f"--test_envs={test_env}",
        f"--output_dir={output_dir}"
    ]

    # Add other arguments if provided
    if args.data_dir:
        cmd.append(f"--data_dir={args.data_dir}")
    if args.task:
        cmd.append(f"--task={args.task}")
    if args.hparams:
        cmd.append(f"--hparams={args.hparams}")
    if args.hparams_seed is not None:
        cmd.append(f"--hparams_seed={args.hparams_seed}")
    if args.trial_seed is not None:
        cmd.append(f"--trial_seed={args.trial_seed}")
    if args.seed is not None:
        cmd.append(f"--seed={args.seed}")
    if args.steps is not None:
        cmd.append(f"--steps={args.steps}")
    if args.checkpoint_freq is not None:
        cmd.append(f"--checkpoint_freq={args.checkpoint_freq}")
    if args.holdout_fraction is not None:
        cmd.append(f"--holdout_fraction={args.holdout_fraction}")
    if args.uda_holdout_fraction is not None:
        cmd.append(f"--uda_holdout_fraction={args.uda_holdout_fraction}")
    if args.skip_model_save:
        cmd.append("--skip_model_save")
    if args.save_model_every_checkpoint:
        cmd.append("--save_model_every_checkpoint")
    if args.init_step:
        cmd.append("--init_step")
    if args.path_for_init != "None":
        cmd.append(f"--path_for_init={args.path_for_init}")
    if args.use_esm:
        cmd.append("--use_esm")

    print(f"\n\n{'='*80}")
    print(f"Running test environment {test_env}")
    print(f"{'='*80}\n")
    print(f"Command: {' '.join(cmd)}")

    # Run the command
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command for test env {test_env}: {e}")
        return False

def combine_results(args, temp_dirs, output_dir):
    """
    Combine results from all runs into a single JSON file
    """
    all_results = {}

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Combine the results.jsonl files
    for i, temp_dir in enumerate(temp_dirs):
        jsonl_path = os.path.join(temp_dir, "results.jsonl")
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r') as f:
                lines = f.readlines()
                results = [json.loads(line) for line in lines]
                all_results[f"test_env_{i}"] = results

            # Copy model files if needed
            if not args.skip_model_save:
                model_src = os.path.join(temp_dir, "model.pkl")
                model_dst = os.path.join(output_dir, f"model_test_env_{i}.pkl")
                if os.path.exists(model_src):
                    shutil.copy(model_src, model_dst)

                best_model_src = os.path.join(temp_dir, "best_model.pkl")
                best_model_dst = os.path.join(output_dir, f"best_model_test_env_{i}.pkl")
                if os.path.exists(best_model_src):
                    shutil.copy(best_model_src, best_model_dst)

    # Write combined results
    with open(os.path.join(output_dir, "combined_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Create a summary of the best results for each test environment
    summary = {}
    for env_key, results in all_results.items():
        if results:
            # Get the best result for this environment (highest agg_test_acc)
            best_result = max(results, key=lambda x: x.get('agg_test_acc', 0))
            summary[env_key] = {
                'test_env': int(env_key.split('_')[-1]),
                'agg_test_acc': best_result.get('agg_test_acc', 0),
                'agg_val_acc': best_result.get('agg_val_acc', 0),
                'step': best_result.get('step', 0),
                'epoch': best_result.get('epoch', 0)
            }

    # Calculate average performance across all environments
    avg_test_acc = sum(item['agg_test_acc'] for item in summary.values()) / len(summary) if summary else 0
    summary['average'] = {
        'avg_test_acc': avg_test_acc
    }

    # Write summary
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    # Also write the summary in a more human-readable format
    with open(os.path.join(output_dir, "summary.txt"), 'w') as f:
        f.write("Summary of results for all test environments\n")
        f.write("="*50 + "\n\n")

        for env_key, result in sorted(summary.items()):
            if env_key == 'average':
                continue
            f.write(f"Test Environment: {result['test_env']}\n")
            f.write(f"Test Accuracy: {result['agg_test_acc']:.4f}\n")
            f.write(f"Validation Accuracy: {result['agg_val_acc']:.4f}\n")
            f.write(f"Step: {result['step']}\n")
            f.write(f"Epoch: {result['epoch']:.2f}\n\n")

        f.write("="*50 + "\n")
        f.write(f"Average Test Accuracy: {summary['average']['avg_test_acc']:.4f}\n")

def main():
    parser = argparse.ArgumentParser(description='Run PDCL on all environments as test')

    # Main parameters
    parser.add_argument('--dataset', type=str, default="AbRosetta",
                        help='Dataset to use')
    parser.add_argument('--algorithm', type=str, default="PDCL",
                        help='Algorithm to use')
    parser.add_argument('--output_dir', type=str, default="./domainbed/outputs/all_envs",
                        help='Output directory for combined results')

    # Optional parameters to pass through to train.py
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory')
    parser.add_argument('--task', type=str, default=None,
                        help='Task type')
    parser.add_argument('--hparams', type=str, default=None,
                        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=None,
                        help='Seed for random hparams')
    parser.add_argument('--trial_seed', type=int, default=None,
                        help='Trial number')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
                        help='Number of steps')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
                        help='Checkpoint every N steps')
    parser.add_argument('--holdout_fraction', type=float, default=None,
                        help='Fraction of data to hold out')
    parser.add_argument('--uda_holdout_fraction', type=float, default=None,
                        help='Fraction of test to use for UDA')
    parser.add_argument('--skip_model_save', action='store_true',
                        help='Skip saving the models')
    parser.add_argument('--save_model_every_checkpoint', action='store_true',
                        help='Save model at every checkpoint')
    parser.add_argument('--init_step', action='store_true',
                        help='Initialize with pre-trained model')
    parser.add_argument('--path_for_init', type=str, default="None",
                        help='Path for initialization')
    parser.add_argument('--use_esm', action='store_true',
                        help='Use ESM embeddings')

    # Parse arguments
    args = parser.parse_args()

    # Print configuration
    print("Running PDCL on all environments as test")
    print(f"Dataset: {args.dataset}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Output directory: {args.output_dir}")

    # Create a unique run ID
    run_id = uuid.uuid4().hex[:8]

    # Create temporary directories for each run
    temp_dirs = []
    for i in range(4):  # Assuming 4 environments (0, 1, 2, 3)
        temp_dir = os.path.join(args.output_dir, f"temp_{run_id}_env_{i}")
        temp_dirs.append(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

    # Run training for each test environment
    success = True
    for i in range(4):  # Assuming 4 environments (0, 1, 2, 3)
        if not run_training(args, i, temp_dirs[i]):
            success = False

    if success:
        # Combine results
        combine_results(args, temp_dirs, args.output_dir)
        print("\nAll runs completed successfully!")
        print(f"Combined results saved to {args.output_dir}/combined_results.json")
        print(f"Summary saved to {args.output_dir}/summary.json and {args.output_dir}/summary.txt")
    else:
        print("\nSome runs failed. Check the logs for details.")

    # Clean up temporary directories (uncomment if you want to delete them)
    # for temp_dir in temp_dirs:
    #     if os.path.exists(temp_dir):
    #         shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
