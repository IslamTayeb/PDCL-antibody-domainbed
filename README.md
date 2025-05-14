# PDCL for Antibody DomainBed: Extending Domain Generalization to Therapeutic Antibodies
This repository extends the [Antibody DomainBed](https://github.com/facebookresearch/DomainBed) framework for benchmarking domain generalization (DG) algorithms on therapeutic antibody sequences.  This research project was conducted under the supervision of Prof. Navid NaderiAlizadeh at Duke University.  The core contribution is the integration of the Primal-Dual Continual Learning (PDCL) algorithm, adapting it for the unique challenges presented by antibody sequence data. You will also find a [Technical Report](https://github.com/IslamTayeb/PDCL-antibody-domainbed/blob/main/domainbed/public/PDCL_for_Antibody_Design.pdf) in the repository.

## Features
* **Antibody Dataset:**  A curated dataset of antibody sequences and binding information, divided into multiple domains. The dataset is based on [In Search of Lost Domain Generalization](https://arxiv.org/abs/2007.01434) and further adapted for antibody sequences. Dataset can be found in the original paper.
* **SeqCNN and ESM Backbones:**  Support for both SeqCNN and ESM (Evolutionary Scale Modeling) architectures as backbones for the antibody sequence processing, enabling comparison of different representation learning approaches.  The choice of backbone is controlled via the `--use_esm` flag in the training script.
* **PDCL Algorithm Implementation:** A comprehensive implementation of the Primal-Dual Continual Learning (PDCL) algorithm, adapted for continual learning in the context of antibody domain generalization.  This includes visualization tools for monitoring dual variables.
* **Dual Variable Visualization:** The PDCL algorithm incorporates functionality to visualize the evolution of dual variables during training. This aids in understanding how the algorithm balances constraints across different domains.
* **Benchmarking Suite:** Provides a complete suite for evaluating various DG algorithms on the antibody dataset, allowing for robust comparison and analysis.
* **Modular Design:** The codebase utilizes a modular design, allowing for easy integration of new algorithms and datasets.

## Theory
The mathematical formulation casts antibody design as a constrained continual learning problem.  At each time step *t*, representing a design cycle with data distribution *D<sub>t</sub>*, the goal is to learn a predictor *f<sub>θ</sub>* that minimizes the loss on the current task while satisfying constraints on previous tasks.  This is expressed as a constrained optimization problem (Equation 1 & 2 in the paper), explicitly balancing plasticity (learning new tasks) and stability (retaining knowledge of past tasks).

The constraints, representing acceptable forgetting on previous tasks, are incorporated into an unconstrained Lagrangian objective function (Equation 3). This Lagrangian introduces dual variables, λ<sub>k</sub>, which act as dynamic penalty weights for violating the constraints on previous tasks (*k* = 1,..., *t*−1).  These dual variables are crucial because they dynamically adjust during training, increasing when constraints are violated and decreasing when satisfied.  This primal-dual approach allows efficient optimization, alternating between updating model parameters *θ* (primal update, Equation 5) and updating dual variables λ<sub>k</sub> (dual update, Equation 6).

Importantly, the optimal dual variables, λ*<sub>k</sub>, provide valuable information. They quantify the sensitivity of the current task's optimal loss (*P<sub>t</sub>*) with respect to the constraint level ε<sub>k</sub> (Equation 7).  A larger λ*<sub>k</sub> indicates a greater sensitivity, signifying that maintaining performance on task *k* is more challenging and requires more attention. This sensitivity information is directly used to adaptively allocate memory in the buffer.

The limited capacity of the memory buffer necessitates an adaptive partitioning strategy.  Instead of uniform allocation, buffer space for each past task *k* is allocated proportionally to its normalized dual variable λ<sub>k</sub>(*t*) (Equation 8).  A parameter α ∈ [0, 1] controls the balance between this dual-variable-driven allocation and a uniform baseline allocation, preventing complete neglect of less challenging tasks.

Finally, feature-based diversity selection (Algorithm 1 in the paper) is used to select a representative subset of examples for each task within the limited buffer space.  This farthest-first approach sequentially adds examples that maximize the minimum distance in feature space to already selected examples, ensuring the buffer stores a diverse range of feature representations.  This diversity-based selection is implemented in `_select_diverse_examples()` utilizing cosine distance on normalized feature vectors, as detailed in the implementation section.  The `update()` method in the `PDCL` class directly implements the primal-dual optimization steps, and the buffer management is handled by `_update_buffer()`, `_partition_buffer()`, and `_fill_buffer()` methods, reflecting the theory described above.

You may find more details regarding the theory within the [Technical Report](https://github.com/IslamTayeb/PDCL-antibody-domainbed/blob/main/domainbed/public/PDCL_for_Antibody_Design.pdf).

## Usage
This project extends the existing DomainBed project.  Before proceeding, ensure you have the original DomainBed repository cloned and the `requirements.txt` file properly configured.

### Training

To train any DomainBed baseline algorithm on the Antibody dataset:

```bash
python -m domainbed.scripts.train --dataset AbRosetta --algorithm ERM --output_dir='./some_directory'
```

To use the ESM backbone instead of SeqCNN:

```bash
python -m domainbed.scripts.train --dataset AbRosetta --algorithm ERM --output_dir='./some_directory' --use_esm
```

Replace `ERM` with any other algorithm supported by DomainBed (see `domainbed/algorithms.py` for a list of available algorithms).  Remember to specify the path to your antibody dataset in `domainbed/datasets.py`.

### PDCL Visualization

The PDCL algorithm automatically logs dual variable data during training. To visualize the dual variables:

1. **Directly from a trained PDCL model:**

```python
# After training a PDCL model
model.visualize_dual_variables(save_path="dual_vars.png", show=True)

# Or to get the raw data for custom visualization
dual_data = model.get_dual_variable_data()
```

2. **From saved checkpoints:**

```bash
python -m domainbed.scripts.visualize_pdcl_duals --input_dir=/path/to/model/checkpoints --output_dir=visualizations --show
```

## Installation
1. Clone the repository: `git clone https://github.com/IslamTayeb/PDCL-antibody-domainbed.git`
2. Navigate to the cloned directory: `cd antibody_domainbed`
3. Install dependencies: `conda create --name <env_name> --file requirements.txt`
4. (Optional) Install `backpack` for Fishr algorithm:  `pip install backpack-for-pytorch==1.3.0`

## Technologies Used
* **PyTorch:**  Deep learning framework used for building and training the models.
* **ESM (Evolutionary Scale Modeling):** Primary protein language model (pLM), used as an alternative backbone to regular seqCNNs.
* **Pandas:** Used for data manipulation and analysis of the antibody dataset.
* **Scikit-learn:** Used for evaluation metrics (precision, recall).
* **Matplotlib:** Used for generating visualizations.
* **Conda:** Environment management tool for handling project dependencies.

## Dependencies
The `requirements.txt` file lists all necessary Python packages.

*README.md was made with [Etchr](https://etchr.dev)*