#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Script to visualize the evolution of dual variables in the PDCL algorithm.
Example usage:
python -m domainbed.scripts.visualize_pdcl_duals \
    --input_dir=/path/to/model/checkpoints \
    --output_dir=/path/to/save/visualizations
"""

import argparse
import os
import sys
import json
import logging

import torch
import matplotlib.pyplot as plt
import numpy as np

from domainbed import algorithms
from domainbed.lib.misc import Tee
from domainbed.lib import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize PDCL dual variables")
    parser.add_argument("--input_dir", type=str, required=True,
        help="Directory containing model checkpoints")
    parser.add_argument("--output_dir", type=str, default="./pdcl_visualizations",
        help="Directory to save the visualizations")
    parser.add_argument("--checkpoint", type=str, default=None,
        help="Specific checkpoint file to visualize (if not specified, visualizes all PDCL checkpoints)")
    parser.add_argument("--show", action="store_true",
        help="Whether to display the plots interactively")
    parser.add_argument("--no_log_file", action="store_true",
        help="Disable writing to a log file (use if you have permission issues)")

    args = parser.parse_args()

    # Try to create output directory
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        has_output_dir_access = True
    except PermissionError:
        print(f"Warning: Cannot create output directory {args.output_dir} due to permission issues.")
        print("Will attempt to save visualizations to current directory instead.")
        args.output_dir = "."
        has_output_dir_access = False

    # Setup logging
    if not args.no_log_file and has_output_dir_access:
        try:
            log_file = os.path.join(args.output_dir, "visualization_log.txt")
            sys.stdout = Tee(log_file, "w")
            print(f"Logging to {log_file}")
        except Exception as e:
            print(f"Warning: Could not create log file: {e}")
            print("Continuing without log file.")

    print(f"Looking for PDCL checkpoints in {args.input_dir}")

    # Find all PDCL checkpoints
    if args.checkpoint:
        checkpoints = [args.checkpoint]
    else:
        checkpoints = []
        try:
            for root, dirs, files in os.walk(args.input_dir):
                for file in files:
                    if file.endswith(".pkl") and "model" in file:
                        checkpoint_path = os.path.join(root, file)
                        try:
                            # Check if this is a PDCL checkpoint
                            checkpoint = torch.load(checkpoint_path)
                            if "args" in checkpoint and checkpoint["args"]["algorithm"] == "PDCL":
                                checkpoints.append(checkpoint_path)
                        except Exception as e:
                            print(f"Warning: Error loading checkpoint {checkpoint_path}: {str(e)}")
        except Exception as e:
            print(f"Error scanning input directory {args.input_dir}: {str(e)}")

    if not checkpoints:
        print(f"No PDCL checkpoints found in {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(checkpoints)} PDCL checkpoints")
    successful_plots = 0

    # Process each checkpoint
    for checkpoint_path in checkpoints:
        try:
            print(f"Processing checkpoint: {checkpoint_path}")

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path)

            # Extract model and its dual variable history
            model_dict = checkpoint["model_dict"]
            args_dict = checkpoint["args"]

            # Check if this is a valid PDCL checkpoint with dual variable history
            if "dual_var_history" not in model_dict:
                print(f"Warning: Checkpoint {checkpoint_path} does not contain dual variable history, skipping")
                continue

            # Extract relevant information
            dual_var_history = model_dict["dual_var_history"]
            iteration_history = model_dict.get("iteration_history", list(range(len(next(iter(dual_var_history.values()), [])))))

            # Create a figure for visualization
            plt.figure(figsize=(12, 8))

            # Plot the evolution of dual variables
            for domain_idx, dual_vars in dual_var_history.items():
                if len(dual_vars) > 0:
                    plt.plot(
                        iteration_history[:len(dual_vars)],
                        dual_vars,
                        label=f'Domain {domain_idx}',
                        marker='o',
                        markersize=3,
                        alpha=0.7
                    )

            # Add styling
            plt.xlabel('Training Iterations')
            plt.ylabel('Dual Variable Value (λ)')
            plt.title(f'Evolution of PDCL Dual Variables: {os.path.basename(checkpoint_path)}')
            plt.legend(loc='best')
            plt.grid(True, linestyle='--', alpha=0.7)

            # Add domain transitions if we can identify them
            domain_steps = args_dict.get("domain_steps", 1000)
            if domain_steps:
                domain_changes = [i for i in range(len(iteration_history)) if i % domain_steps == 0]
                for i, change_idx in enumerate(domain_changes):
                    if change_idx < len(iteration_history):
                        change = iteration_history[change_idx]
                        plt.axvline(x=change, color='gray', linestyle='--', alpha=0.5)
                        plt.annotate(
                            f'Domain {i} → {i+1}',
                            xy=(change, 0),
                            xytext=(change, -0.02),
                            textcoords='data',
                            ha='center',
                            va='top',
                            rotation=90,
                            fontsize=8
                        )

            # Save the visualization
            try:
                output_filename = os.path.join(
                    args.output_dir,
                    f"pdcl_dual_variables_{os.path.basename(checkpoint_path).replace('.pkl', '.png')}"
                )
                plt.savefig(output_filename, dpi=300, bbox_inches='tight')
                print(f"Saved visualization to {output_filename}")
                successful_plots += 1
            except Exception as e:
                print(f"Error saving visualization: {str(e)}")
                # Try saving to current directory as fallback
                try:
                    fallback_filename = f"pdcl_dual_variables_{os.path.basename(checkpoint_path).replace('.pkl', '.png')}"
                    plt.savefig(fallback_filename, dpi=300, bbox_inches='tight')
                    print(f"Saved visualization to current directory: {fallback_filename}")
                    successful_plots += 1
                except Exception as e2:
                    print(f"Could not save visualization even to current directory: {str(e2)}")

            # Show the plot if requested
            if args.show:
                plt.show()
            else:
                plt.close()

            # Also save the raw data as JSON for further analysis
            try:
                json_output = os.path.join(
                    args.output_dir,
                    f"pdcl_dual_variables_{os.path.basename(checkpoint_path).replace('.pkl', '.json')}"
                )

                # Convert tensor keys to strings for JSON serialization
                serializable_data = {
                    'iterations': iteration_history,
                    'dual_variables': {str(k): v for k, v in dual_var_history.items()}
                }

                with open(json_output, 'w') as f:
                    json.dump(serializable_data, f, indent=2)

                print(f"Saved raw data to {json_output}")
            except Exception as e:
                print(f"Error saving JSON data: {str(e)}")
                # Try saving to current directory as fallback
                try:
                    fallback_json = f"pdcl_dual_variables_{os.path.basename(checkpoint_path).replace('.pkl', '.json')}"
                    with open(fallback_json, 'w') as f:
                        json.dump(serializable_data, f, indent=2)
                    print(f"Saved JSON data to current directory: {fallback_json}")
                except Exception as e2:
                    print(f"Could not save JSON data even to current directory: {str(e2)}")

        except Exception as e:
            print(f"Error processing checkpoint {checkpoint_path}: {str(e)}")

    print(f"Visualization complete! Successfully generated {successful_plots} plots.")
