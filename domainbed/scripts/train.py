# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt
from functools import partial

import numpy as np
import PIL
import torch
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed.lib.collate import rn_collate, esm_collate, pad_x,Alphabet, PaddCollator
from domainbed.lib.logger import create_logger
import warnings
from torch.serialization import SourceChangeWarning
logger = create_logger(__name__)

warnings.filterwarnings('always')


warnings.filterwarnings("ignore", category=SourceChangeWarning)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str, default=".")
    parser.add_argument('--dataset', type=str, default="Ab")
    parser.add_argument('--algorithm', type=str, default="IRM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N ste1ps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="./")
    parser.add_argument('--target', type=str, default="None")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--init_step', action='store_true')
    parser.add_argument('--path_for_init', type=str, default="None")
    parser.add_argument('--use_esm', action='store_true')
    parser.add_argument('--viz', action='store_true',
                      help='Run multiple experiments with different hyperparameters to generate comprehensive visualizations')
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset, args.use_esm)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset, args.use_esm,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print("Device: ", device)

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](
            args.data_dir,
            args.test_envs, hparams, args.target, use_esm=args.use_esm)
        logger.info("Built dataset")
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []

    for env_i, env in enumerate(dataset):
        uda = []
        logger.info(f"Constructing environment {env_i}")

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))
    logger.info("Done with in/out splits")

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    #############
    # Algorithm #
    #############
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    logger.info(f"Algorithm: {args.algorithm}")
    if args.algorithm == "ERM":
        algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
                                    len(dataset) - len(args.test_envs), hparams,
                                    init_step=args.init_step,
                                    path_for_init=args.path_for_init)
    else:
        algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
                                    len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)
    algorithm.to(device)

    ###########
    # Loaders #
    ###########
    # Pass in custom collate function from featurizer's own tokenizer if ESM
    if args.use_esm:
        try:
            collate_fn = partial(
                esm_collate, x_collate_fn=algorithm.featurizer.get_batch_tensor_x)
        except:
            collate_fn = partial(
                esm_collate, x_collate_fn=algorithm.network.featurizer.get_batch_tensor_x)
    else:
        collate_fn = rn_collate
    logger.info(f"Batch size: {hparams['batch_size']}")
    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS,
        collate_fn=collate_fn)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]
    logger.info("Set train dataloader")

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS,
        collate_fn=collate_fn)
        for i, (env, env_weights) in enumerate(uda_splits)]
    logger.info("Set uda dataloader")

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS,
        collate_fn=collate_fn)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    def save_checkpoint(filename, results=None):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        if results is not None:
            save_dict["results"] = results
        torch.save(save_dict, os.path.join(args.output_dir, filename))


    last_results_keys = None
    best_acc = 0
    for step in range(start_step, n_steps):
        logger.debug(f"Update step: {step}")
        step_start_time = time.time()
        minibatches_device = [(x, y)
            for x, y in next(train_minibatches_iterator)]
        if args.use_esm:
            try:
                minibatches_device = pad_x(
                    minibatches_device,
                    padding_idx=algorithm.featurizer.alphabet.padding_idx)
            except:
                minibatches_device = pad_x(
                    minibatches_device,
                    padding_idx=algorithm.network.featurizer.alphabet.padding_idx)
        minibatches_device = [(x.to(device), y.to(device))
            for x, y in minibatches_device]
        if args.task == "domain_adaptation":
            uda_device = [x.to(device) for x, _ in next(uda_minibatches_iterator)]
            if args.use_esm:
                try:
                    uda_device = pad_x(
                        uda_device,
                        padding_idx=algorithm.featurizer.alphabet.padding_idx)
                except:
                    uda_device = pad_x(
                        uda_device,
                        padding_idx=algorithm.network.featurizer.alphabet.padding_idx)
        else:
            uda_device = None
        step_vals = algorithm.update(minibatches_device, uda_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                acc = misc.accuracy(algorithm, loader, weights, device)
                results[name+'_acc'] = acc
                # feats, gt_labels, preds = misc.features(algorithm, loader, weights, device)
                # torch.save({'features': feats, 'labels': gt_labels, 'preds': preds},
                #     os.path.join(args.output_dir, 'output.pt'))

            ##========
            agg_val_acc, nagg_val_acc = 0, 0
            for name in results.keys():
                if 'acc' in name and 'out' in name and int(name.split('env')[1].split('_')[0]) not in args.test_envs:
                    agg_val_acc += results[name]
                    nagg_val_acc += 1.
            agg_val_acc /= (nagg_val_acc + 1e-9)
            results['agg_val_acc'] = agg_val_acc

            agg_test_acc, nagg_test_acc = 0, 0
            for name in results.keys():
                if 'acc' in name and name !='agg_val_acc' and int(name.split('env')[1].split('_')[0]) in args.test_envs:
                    agg_test_acc += results[name]
                    nagg_test_acc += 1.
            agg_test_acc /= (nagg_test_acc + 1e-9)
            results['agg_test_acc'] = agg_test_acc
            ##========

            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')

            if best_acc < agg_val_acc:
                # logger.info(f'Saving best model... at update step {step}')
                best_acc = agg_val_acc
                save_checkpoint('best_model.pkl', results=json.dumps(results, sort_keys=True))

    if args.init_step:
        algorithm.save_path_for_future_init(args.path_for_init)
    save_checkpoint('model.pkl')

    # Special visualization mode: run multiple experiments with different hyperparameters
    if args.viz and args.algorithm == "PDCL":
        print("Visualization mode activated! Running multiple experiments to generate comprehensive visualizations...")

        # Create visualization directories
        viz_dir = os.path.join(args.output_dir, 'comprehensive_visualizations')
        os.makedirs(viz_dir, exist_ok=True)

        # Base hyperparameters from the current run
        base_hparams = hparams.copy()

        # For tracking experiment results
        epsilon_results = {}

        # Save original algorithm for comparison and as baseline
        base_algorithm = algorithm
        baseline_epsilon = hparams['epsilon']

        # 1. Run experiments with different epsilon values (5000 steps each)
        epsilon_values = [0.01, 0.03, 0.05, 0.08, 0.12, 0.2]
        viz_steps = 5000  # Shorter run for visualization experiments

        print(f"Running {len(epsilon_values)} experiments with different epsilon values...")
        for i, eps in enumerate(epsilon_values):
            if eps == baseline_epsilon:
                # Skip rerunning baseline (use original results)
                print(f"Using baseline results for epsilon = {eps}")
                epsilon_results[eps] = {'stability': None, 'plasticity': None, 'overall': None}

                # Extract performance metrics from base_algorithm
                if hasattr(base_algorithm, 'stability_metrics') and len(base_algorithm.stability_metrics) > 0:
                    epsilon_results[eps]['stability'] = base_algorithm.stability_metrics[-1]
                if hasattr(base_algorithm, 'plasticity_metrics') and len(base_algorithm.plasticity_metrics) > 0:
                    epsilon_results[eps]['plasticity'] = base_algorithm.plasticity_metrics[-1]
                if hasattr(base_algorithm, 'overall_metrics') and len(base_algorithm.overall_metrics) > 0:
                    epsilon_results[eps]['overall'] = base_algorithm.overall_metrics[-1]
                continue

            print(f"\nExperiment {i+1}/{len(epsilon_values)}: Running with epsilon = {eps}")

            # Create experiment-specific hparams
            exp_hparams = base_hparams.copy()
            exp_hparams['epsilon'] = eps

            # Create a new algorithm instance with the experimental epsilon
            algorithm_class = algorithms.get_algorithm_class(args.algorithm)
            exp_algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
                                          len(dataset) - len(args.test_envs), exp_hparams)
            exp_algorithm.to(device)

            # Train for viz_steps (limited training for visualization)
            train_minibatches_iterator = zip(*train_loaders)
            for step in range(viz_steps):
                if step % 500 == 0:
                    print(f"  Step {step}/{viz_steps}...")

                minibatches_device = [(x, y) for x, y in next(train_minibatches_iterator)]
                if args.use_esm:
                    try:
                        minibatches_device = pad_x(
                            minibatches_device,
                            padding_idx=exp_algorithm.featurizer.alphabet.padding_idx)
                    except:
                        minibatches_device = pad_x(
                            minibatches_device,
                            padding_idx=exp_algorithm.network.featurizer.alphabet.padding_idx)

                minibatches_device = [(x.to(device), y.to(device)) for x, y in minibatches_device]
                exp_algorithm.update(minibatches_device, None)

            # Evaluate on a larger test batch to get more accurate metrics
            print(f"  Evaluating final performance for epsilon = {eps}...")

            # Extract the metrics from the last 5 steps (more stable than just the last one)
            stability_values = []
            plasticity_values = []
            overall_values = []

            # Get metrics from the algorithm's tracking
            if hasattr(exp_algorithm, 'stability_metrics') and len(exp_algorithm.stability_metrics) >= 5:
                stability_values = exp_algorithm.stability_metrics[-5:]
            if hasattr(exp_algorithm, 'plasticity_metrics') and len(exp_algorithm.plasticity_metrics) >= 5:
                plasticity_values = exp_algorithm.plasticity_metrics[-5:]
            if hasattr(exp_algorithm, 'overall_metrics') and len(exp_algorithm.overall_metrics) >= 5:
                overall_values = exp_algorithm.overall_metrics[-5:]

            # Use averages of last 5 values for stability
            epsilon_results[eps] = {
                'stability': sum(stability_values) / max(len(stability_values), 1) if stability_values else None,
                'plasticity': sum(plasticity_values) / max(len(plasticity_values), 1) if plasticity_values else None,
                'overall': sum(overall_values) / max(len(overall_values), 1) if overall_values else None
            }

            # Verify and log collected metrics
            print(f"  Metrics collected for epsilon = {eps}:")
            print(f"    Stability: {epsilon_results[eps]['stability']}")
            print(f"    Plasticity: {epsilon_results[eps]['plasticity']}")
            print(f"    Overall: {epsilon_results[eps]['overall']}")

            # Clean up to free memory
            del exp_algorithm
            torch.cuda.empty_cache()

        # 2. Now create a new visualization for domain transitions
        print("\nGenerating domain transition visualization...")

        # Use the baseline algorithm's data to create a domain transition visualization
        if hasattr(base_algorithm, 'iteration_history') and hasattr(base_algorithm, 'dual_var_history'):
            try:
                # Create domain transition visualization
                transition_viz_path = os.path.join(viz_dir, 'domain_transition_viz.png')

                # Call a new method to visualize domain transitions
                base_algorithm.visualize_domain_transitions(save_path=transition_viz_path, show=False)
                print(f"Domain transition visualization saved to {transition_viz_path}")
            except Exception as e:
                print(f"Warning: Could not generate domain transition visualization: {str(e)}")

        # 3. Now let's create the comprehensive constraint impact visualization
        print("\nGenerating comprehensive visualizations...")

        # Create a helper function to convert results to the format expected by visualization methods
        def prepare_epsilon_data_for_visualization(results):
            # Validate and convert results into the format expected by the visualization methods
            data = {}

            # Check for valid entries and filter out None values
            valid_results = {k: v for k, v in results.items()
                            if v['stability'] is not None and
                               v['plasticity'] is not None and
                               v['overall'] is not None}

            if not valid_results:
                print("Warning: No valid data found for constraint impact visualization")
                return {}

            for eps, metrics in valid_results.items():
                # Validate metrics are in reasonable range (0-1)
                stability = min(max(metrics['stability'], 0.0), 1.0)
                plasticity = min(max(metrics['plasticity'], 0.0), 1.0)
                overall = min(max(metrics['overall'], 0.0), 1.0)

                data[eps] = {
                    'epsilon': eps,
                    'stability': {'values': [stability], 'mean': stability, 'std': 0},
                    'plasticity': {'values': [plasticity], 'mean': plasticity, 'std': 0},
                    'overall': {'values': [overall], 'mean': overall, 'std': 0},
                    'iterations': [viz_steps]
                }
            return data

        # Save the data
        with open(os.path.join(viz_dir, 'epsilon_results.json'), 'w') as f:
            json.dump({str(k): v for k, v in epsilon_results.items()}, f, indent=2)

        # Replace the algorithm's internal data structures with our comprehensive results
        algorithm.epsilon_performances = prepare_epsilon_data_for_visualization(epsilon_results)

        # Generate the constraint impact visualization
        constraint_path = os.path.join(viz_dir, 'comprehensive_constraint_impact.png')
        algorithm.visualize_constraint_impact(save_path=constraint_path, show=False)
        print(f"Comprehensive constraint impact visualization saved to {constraint_path}")

        # Generate the dual variables visualization for completeness
        dual_path = os.path.join(viz_dir, 'dual_variables.png')
        algorithm.visualize_dual_variables(save_path=dual_path, show=False)
        print(f"Dual variables visualization saved to {dual_path}")

        print("\nVisualization mode complete! All visualizations saved to:", viz_dir)

    # Generate visualization for PDCL if applicable (standard case)
    elif args.algorithm == "PDCL":
        try:
            print("Generating PDCL visualizations...")
            visualization_dir = os.path.join(args.output_dir, 'visualizations')
            os.makedirs(visualization_dir, exist_ok=True)

            # Dual variables visualization
            visualization_path = os.path.join(visualization_dir, 'dual_variables.png')
            algorithm.visualize_dual_variables(save_path=visualization_path, show=False)
            print(f"PDCL dual variables visualization saved to {visualization_path}")

            # Constraint impact visualization
            constraint_path = os.path.join(visualization_dir, 'constraint_impact.png')
            algorithm.visualize_constraint_impact(save_path=constraint_path, show=False)
            print(f"PDCL constraint impact visualization saved to {constraint_path}")

            # Buffer impact visualization
            buffer_path = os.path.join(visualization_dir, 'buffer_impact.png')
            algorithm.visualize_buffer_impact(save_path=buffer_path, show=False)
            print(f"PDCL buffer impact visualization saved to {buffer_path}")

            # Save raw data
            data_path = os.path.join(visualization_dir, 'dual_variables_data.json')
            dual_data = algorithm.get_dual_variable_data()

            # Convert data to serializable format
            serializable_data = {
                'iterations': dual_data['iterations'],
                'dual_variables': {str(k): v for k, v in dual_data['dual_variables'].items()}
            }

            with open(data_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            print(f"PDCL dual variable data saved to {data_path}")

            # Create an HTML index file to organize and explain the visualizations
            index_path = os.path.join(visualization_dir, 'index.html')
            html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDCL Algorithm Visualizations</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .visualization-container {{
            margin-bottom: 40px;
            padding: 15px;
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            background-color: #f9f9f9;
        }}
        .visualization-image {{
            width: 100%;
            max-width: 1000px;
            margin: 20px 0;
            border: 1px solid #ccc;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .description {{
            margin-bottom: 15px;
        }}
        .code {{
            font-family: monospace;
            background-color: #f5f5f5;
            padding: 2px 4px;
            border-radius: 3px;
        }}
        .info-box {{
            background-color: #e8f4f8;
            border-left: 4px solid #5bc0de;
            padding: 10px 15px;
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    <h1>PDCL Algorithm Visualizations</h1>
    <p>This page presents visualizations of the Primal-Dual Continual Learning (PDCL) algorithm's performance on the Antibody DomainBed benchmark. These visualizations help understand the algorithm's behavior and the impact of its key hyperparameters.</p>

    <div class="visualization-container">
        <h2>Dual Variables Evolution</h2>
        <div class="description">
            <p>This visualization shows how dual variables (λ) evolve during training. Dual variables are Lagrange multipliers that enforce constraints on the loss of previously learned domains. Higher values indicate greater difficulty in satisfying constraints.</p>
            <div class="info-box">
                <strong>Interpretation:</strong> Spikes in dual variables indicate challenging transitions between domains. As the model adapts, dual variables typically decrease, showing that constraints are being satisfied.
            </div>
        </div>
        <img src="dual_variables.png" alt="Dual Variables Evolution" class="visualization-image">
    </div>

    <div class="visualization-container">
        <h2>Constraint Level (ε) Impact</h2>
        <div class="description">
            <p>This visualization shows how different constraint levels (ε) affect the trade-off between stability (performance on previous domains) and plasticity (performance on current domain).</p>
            <div class="info-box">
                <strong>Interpretation:</strong> The optimal ε value balances stability and plasticity. Too small ε leads to forgetting previous domains (low stability), while too large ε restricts learning new domains (low plasticity).
            </div>
        </div>
        <img src="constraint_impact.png" alt="Constraint Level Impact" class="visualization-image">
    </div>

    <div class="visualization-container">
        <h2>Buffer Size Impact</h2>
        <div class="description">
            <p>This visualization shows how buffer size affects performance and memory usage. The buffer stores examples from previous domains for replay-based constraint enforcement.</p>
            <div class="info-box">
                <strong>Interpretation:</strong> Larger buffers typically improve performance but with diminishing returns and increased memory requirements. The visualization helps identify the optimal buffer size for your computational resources.
            </div>
        </div>
        <img src="buffer_impact.png" alt="Buffer Size Impact" class="visualization-image">
    </div>

    <div class="info-box">
        <h3>Raw Data</h3>
        <p>The raw data for the dual variables visualization is available as JSON at <a href="dual_variables_data.json">dual_variables_data.json</a> for further analysis.</p>
    </div>

    <footer>
        <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} during PDCL training</p>
    </footer>
</body>
</html>'''

            try:
                with open(index_path, 'w') as f:
                    f.write(html_content)
                print(f"Visualization index created at {index_path}")
            except Exception as e:
                print(f"Warning: Failed to create visualization index: {str(e)}")

        except Exception as e:
            print(f"Warning: Failed to generate PDCL visualizations: {str(e)}")
            # Try fallback visualization
            try:
                algorithm.visualize_dual_variables(save_path="pdcl_dual_variables.png", show=False)
                algorithm.visualize_constraint_impact(save_path="pdcl_constraint_impact.png", show=False)
                algorithm.visualize_buffer_impact(save_path="pdcl_buffer_impact.png", show=False)
                print("PDCL visualizations saved to current directory")
            except Exception as e2:
                print(f"Warning: Fallback visualizations also failed: {str(e2)}")

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')

    # Properly close output redirection
    try:
        if hasattr(sys.stdout, 'close'):
            sys.stdout.close()
        if hasattr(sys.stderr, 'close'):
            sys.stderr.close()
    except Exception as e:
        print(f"Warning: Error while closing streams: {str(e)}", file=sys.__stdout__)
