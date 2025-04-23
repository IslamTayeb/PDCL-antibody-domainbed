# Welcome to Antibody DomainBed

DomainBed is a PyTorch suite containing benchmark datasets and algorithms for domain generalization, as introduced in [In Search of Lost Domain Generalization](https://arxiv.org/abs/2007.01434).

We extend this repo to allow for benchmarking DG algorithms for biological sequences, namely, therapeutic antibodies.
To do so, we adjust the backbones to SeqCNN or ESM, whcih is specified  by adding the `--is_esm` flag to the train script.

## Dataset
Our dataset can be found here:
Before running any tests, please make sure you change the path in domainbed/datasets.py to whereever you store the data

## Quickstart
Set up an environment with all necessary packages `conda create --name <env_name> --file requirements.txt`
Train any DG baseline from Domainbed on the Antibody datset as follows:
`python -m domainbed.scripts.train --dataset AbRosetta --algorithm ERM --output_dir='./some_directory'`

Dataset is available under `domainbed/data/antibody_domainbed_dataset.csv`

All other instructions from the main [Domainbed repo](https://github.com/facebookresearch/DomainBed) hold, please see the original repo for more details on running sweeps.

## PDCL Algorithm with Dual Variable Visualization

The Primal-Dual Continual Learning (PDCL) algorithm now includes functionality to track and visualize the evolution of dual variables across training iterations. This provides insights into how the algorithm is balancing constraints across different domains.

### Using the Visualization Feature

1. **During training**: The PDCL algorithm automatically tracks dual variable evolution during training. The data is stored in the model checkpoints.

2. **Directly from a PDCL model**:
   ```python
   # After training a PDCL model
   model.visualize_dual_variables(save_path="dual_vars.png", show=True)

   # Or to get the raw data for custom visualization
   dual_data = model.get_dual_variable_data()
   ```

3. **From saved checkpoints**:
   ```bash
   python -m domainbed.scripts.visualize_pdcl_duals \
       --input_dir=/path/to/model/checkpoints \
       --output_dir=visualizations \
       --show
   ```

   This script will find all PDCL checkpoints in the input directory, visualize the dual variable evolution for each, and save the plots to the output directory.

### Interpreting the Visualization

The plots show:
- Evolution of dual variables for each domain over training iterations
- Vertical lines indicating domain transitions
- Higher dual variable values indicate domains with more challenging constraints

This visualization can help understand how the algorithm balances performance across domains and how the constraint importance changes during training.
