"""Default hyperparameters. A run overrides them with a JSON dict (--hparams)."""

SEQUENTIAL = ('Finetune', 'ER', 'ER_FF', 'PDCL_RS', 'PDCL')

ALGORITHM_HPARAMS = {
    'IRM': {'irm_lambda': 1e2, 'irm_penalty_anneal_iters': 500},
    'VREx': {'vrex_lambda': 1e1, 'vrex_penalty_anneal_iters': 500},
    'GroupDRO': {'groupdro_eta': 1e-2},
    'CORAL': {'mmd_gamma': 1.},
    'SD': {'sd_reg': 0.1},
}

SEQUENTIAL_HPARAMS = {
    'primal_lr': 5e-5,
    'dual_lr': 0.01,
    'buffer_size': 200,       # stored examples in total
    'pool_size': 1000,        # reservoir of the current round
    'epsilon_scale': 1.1,     # epsilon_k = epsilon_scale * loss reached on round k; 0 = fixed epsilon
    'epsilon': 0.05,
    'alpha': 0.5,             # weight of the dual variables in the memory partition
    'domain_steps': 0,        # 0 = split the training steps equally over the rounds
}


def default_hparams(algorithm):
    hparams = {'lr': 5e-5, 'weight_decay': 0., 'batch_size': 32}
    hparams.update(ALGORITHM_HPARAMS.get(algorithm, {}))
    if algorithm in SEQUENTIAL:
        hparams.update(SEQUENTIAL_HPARAMS)
    return hparams
