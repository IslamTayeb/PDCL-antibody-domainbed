# Run All Environments Script

This script automates running the PDCL algorithm (or any other DomainBed algorithm) on all environments as test environments and combines the results.

## Usage

Basic usage:
```bash
python -m domainbed.scripts.run_all_envs --dataset AbRosetta --algorithm PDCL --output_dir='./domainbed/outputs/my_experiment'
```

This will:
1. Run 4 separate training processes, each using one of environments 0, 1, 2, or 3 as the test environment
2. Save individual results in temporary directories
3. Combine results into a single JSON file in the specified output directory
4. Generate a summary of the best results for each test environment

## Parameters

The script accepts all the same parameters as the regular training script:

```bash
python -m domainbed.scripts.run_all_envs \
  --dataset AbRosetta \
  --algorithm PDCL \
  --output_dir='./domainbed/outputs/my_experiment' \
  --data_dir='/path/to/data' \
  --hparams='{"primal_lr": 1e-4, "dual_lr": 0.01, "epsilon": 0.05}' \
  --steps=5000 \
  --checkpoint_freq=1000
```

## Output Files

The script generates several output files:

1. **combined_results.json**: Contains all the results from all runs organized by test environment
2. **summary.json**: Contains a summary of the best results for each test environment
3. **summary.txt**: A more human-readable format of the summary
4. **model_test_env_X.pkl**: The final model for test environment X
5. **best_model_test_env_X.pkl**: The best model for test environment X

## Extended Example

If you want to run a more comprehensive experiment, you can do something like:

```bash
python -m domainbed.scripts.run_all_envs \
  --dataset AbRosetta \
  --algorithm PDCL \
  --output_dir='./domainbed/outputs/pdcl_experiment' \
  --hparams='{"primal_lr": 5e-5, "dual_lr": 0.01, "epsilon": 0.05, "buffer_size": 200}' \
  --steps=10000 \
  --checkpoint_freq=1000 \
  --seed=0
```

This will run the experiment with specific hyperparameters and save the results.
