"""Train many independent runs in one process.

Every run of this study uses the same small network, so M runs can be trained
as one model with a leading "run" dimension: the convolutions become grouped
convolutions and each optimizer step updates all M runs at once. The runs stay
independent (separate weights, data order, memory, dual variables); only the
arithmetic is shared.

Semantics follow domainbed/scripts/train.py and domainbed/algorithms.py (ERM
and the ER family: Finetune, ER, ER_FF, PDCL_RS, PDCL). Initial weights, data
splits and the minibatch stream of each run are the same as in train.py, and
results are written in the same layout, so the analysis scripts read them
unchanged. Floating-point results are not bit-identical to train.py.

    python -m domainbed.scripts.train_batched --algorithms ERM PDCL --seeds 0 1 2
    python -m domainbed.scripts.train_batched --spec sweeps.json
"""

import argparse
import contextlib
import hashlib
import io
import itertools
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

from domainbed import algorithms, datasets, hparams_registry, streams
from domainbed.lib import misc

REPO_ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_FRACTION = 0.2
AB_LEN = 298                       # heavy + light chain positions
RECENT = 200                       # losses averaged for the constraint level
ERM_FAMILY = ('ERM', 'IRM', 'VREx', 'GroupDRO', 'CORAL', 'SD')   # joint training
PARAMS = ('emb_ab', 'emb_ag', 'c1w', 'c1b', 'c2w', 'c2b', 'fw', 'fb', 'cw', 'cb')


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_tokens(data_dir, device='cpu'):
    """Tokens and labels of the whole dataset in round order (cached on disk)."""
    cache = Path(data_dir) / 'abrosetta_tokens.pt'
    if cache.exists():
        d = torch.load(cache)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            ds = datasets.AbRosetta(data_dir)
        toks, ys, sizes = [], [], []
        for env in ds:
            loader = torch.utils.data.DataLoader(
                env, batch_size=256, num_workers=4, collate_fn=misc.collate)
            xs, labels = zip(*loader)
            x, y = torch.cat(xs), torch.cat(labels)
            assert torch.equal(x, x.to(torch.uint8).float())
            # The network reads each position as the argmax of its one-hot row.
            toks.append(torch.cat([c.to(device).argmax(2).to(torch.uint8).cpu()
                                   for c in x.split(1024)]))
            ys.append(y)
            sizes.append(len(env))
        d = {'tok': torch.cat(toks), 'y': torch.cat(ys), 'sizes': sizes}
        torch.save(d, cache)
    return d


def load_data(data_dir, device, stream='rounds', d=None):
    """The dataset on the device, grouped into the tasks of the given stream."""
    d = d or load_tokens(data_dir, device)
    if stream != 'rounds':
        order, sizes = streams.task_order(data_dir, stream)
        order = torch.as_tensor(order)
        d = {'tok': d['tok'][order], 'y': d['y'][order], 'sizes': sizes}
        rate = [round(float(c.argmax(1).float().mean()), 3) for c in d['y'].split(sizes)]
        print(f"stream {stream}: sizes {sizes}, binder rate {rate}", flush=True)
    sizes = d['sizes']
    tok = d['tok'].to(device)
    data = {
        'tok': tok, 'y': d['y'].to(device), 'sizes': sizes,
        'offsets': np.concatenate([[0], np.cumsum(sizes)[:-1]]),
        'env': torch.cat([torch.full((n,), e) for e, n in enumerate(sizes)]).to(device),
        'input_shape': (tok.size(1), 22),
    }
    # Antigens repeat (a few hundred distinct ones), so at evaluation their
    # branch of the network is computed once per distinct antigen.
    data['ag_unique'], data['ag_inverse'] = torch.unique(
        tok[:, AB_LEN:], dim=0, return_inverse=True)
    # Token 0 is both padding and a residue that embeds to zero, so trailing
    # zeros are padding as far as the network is concerned.
    def true_length(t):
        positions = torch.arange(1, t.size(1) + 1, device=t.device)
        return ((t != 0) * positions).amax(1)
    data['ag_len'] = true_length(tok[:, AB_LEN:])
    data['ag_unique_len'] = true_length(data['ag_unique'])
    return data


def split_keys(n, trial_seed, env_i):
    """The in/out split of train.py: (in keys, out keys), in loader order."""
    keys = list(range(n))
    np.random.RandomState(misc.seed_hash(trial_seed, env_i)).shuffle(keys)
    n_out = int(n * HOLDOUT_FRACTION)
    return keys[n_out:], keys[:n_out]


# --------------------------------------------------------------------------
# the network, with a leading run dimension on every parameter
# --------------------------------------------------------------------------
def branch_reference(P, name, tok):
    """The network as written in networks.py, with a run dimension: embedding,
    two convolutions, max-pooling. tok: [M, B, L] -> [M, B, 128]."""
    M, B, L = tok.shape
    emb = P[name]
    offset = torch.arange(M, device=tok.device).view(M, 1, 1) * emb.size(1)
    e = F.embedding(tok.long() + offset, emb.reshape(-1, emb.size(2)))     # [M, B, L, C]
    h = e.permute(1, 0, 3, 2).reshape(B, M * emb.size(2), L)
    h = F.relu(F.conv1d(h, P['c1w'].flatten(0, 1), P['c1b'].flatten(), stride=2, groups=M))
    h = F.relu(F.conv1d(h, P['c2w'].flatten(0, 1), P['c2b'].flatten(), stride=2, groups=M))
    return torch.max(h, dim=2)[0].view(B, M, -1).transpose(0, 1)


def branch(P, name, tok):
    """Same function as branch_reference, computed faster.

    The input of the first convolution is an embedding of discrete tokens, so
    its output at a window is a sum over the window's offsets of entries of a
    table T[offset, token] = W[:, :, offset] @ embedding[token]. Building the
    table costs 22 * 10 products per run instead of one per position.
    """
    M, B, L = tok.shape
    emb, w1 = P[name], P['c1w']
    K, V, C = w1.size(3), emb.size(1), w1.size(1)
    table = torch.einsum('mock,mtc->mkto', w1, emb)                        # [M, K, V, C]
    windows = tok.long().unfold(2, K, 2)                                   # [M, B, L1, K]
    L1 = windows.size(2)
    index = (windows + torch.arange(K, device=tok.device) * V
             + torch.arange(M, device=tok.device).view(M, 1, 1, 1) * (K * V))
    h = F.embedding_bag(index.reshape(-1, K), table.reshape(-1, C), mode='sum')
    h = F.relu(h.view(M, B, L1, C) + P['c1b'][:, None, None])
    h = h.permute(1, 0, 3, 2).reshape(B, M * C, L1)
    h = F.relu(F.conv1d(h, P['c2w'].flatten(0, 1), P['c2b'].flatten(), stride=2, groups=M))
    return torch.max(h, dim=2)[0].view(B, M, -1).transpose(0, 1)


# A second-layer output sees 18 input positions, in steps of 4. Keeping 21
# positions beyond the longest sequence leaves at least one output that sees
# padding only, so max-pooling over the trimmed input equals max-pooling over
# the full one.
TRIM_MARGIN = 21


def antigen_branch(P, tok, length, fn=branch):
    """branch() on antigens, without convolving over most of the padding.

    Antigens are padded to 915 positions and average about 300. The examples of
    each run are sorted by length, so that position b holds every run's b-th
    longest antigen, and each slice of positions is trimmed to its longest one.
    tok: [M, B, L], length: [M, B] (positions up to the last non-padding token).
    """
    M, B, L = tok.shape
    length, order = length.sort(dim=1, descending=True)
    tok = tok.gather(1, order.unsqueeze(2).expand(-1, -1, L))
    longest = length.max(0).values.tolist()
    cuts = sorted({0, B // 16, B // 8, B // 4, B // 2, B})
    out = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        keep = longest[lo] + TRIM_MARGIN
        out.append(fn(P, 'emb_ag', tok[:, lo:hi, :(L if keep > L else keep)]))
    out = torch.cat(out, dim=1)
    return out.gather(1, order.argsort(1).unsqueeze(2).expand(-1, -1, out.size(2)))


def head(P, h_ab, h_ag):
    feats = torch.baddbmm(P['fb'].unsqueeze(1), torch.cat([h_ab, h_ag], dim=2),
                          P['fw'].transpose(1, 2))
    return feats, torch.baddbmm(P['cb'].unsqueeze(1), feats, P['cw'].transpose(1, 2))


def forward(P, data, rows):
    """rows: [M, B] example indices -> (features [M, B, 256], logits [M, B, 2])."""
    tok = data['tok'][rows]
    return head(P, branch(P, 'emb_ab', tok[:, :, :AB_LEN]),
                antigen_branch(P, tok[:, :, AB_LEN:], data['ag_len'][rows]))


def forward_reference(P, data, rows):
    tok = data['tok'][rows]
    return head(P, branch_reference(P, 'emb_ab', tok[:, :, :AB_LEN]),
                branch_reference(P, 'emb_ag', tok[:, :, AB_LEN:]))


@torch.no_grad()
def forward_all(P, data, rows_per_chunk=20000):
    """Logits of every run on every example of the dataset: [M, N, 2]."""
    M = P['cb'].size(0)
    uniq, uniq_len = data['ag_unique'], data['ag_unique_len']
    step = max(16, rows_per_chunk // (2 * M))
    h_ag = torch.cat([antigen_branch(P, uniq[i:i + step].expand(M, -1, -1),
                                     uniq_len[i:i + step].expand(M, -1))
                      for i in range(0, uniq.size(0), step)], dim=1)
    step = max(1, rows_per_chunk // M)
    out = []
    for i in range(0, data['tok'].size(0), step):
        tok = data['tok'][i:i + step, :AB_LEN].expand(M, -1, -1)
        out.append(head(P, branch(P, 'emb_ab', tok),
                        h_ag[:, data['ag_inverse'][i:i + step]])[1])
    return torch.cat(out, dim=1)


@torch.no_grad()
def features_of(P, j, data, rows):
    """Normalized features of run j on the given examples (for farthest-first)."""
    Pj = {k: v[j:j + 1] for k, v in P.items()}
    feats = torch.cat([forward(Pj, data, chunk.unsqueeze(0))[0][0] for chunk in rows.split(256)])
    return F.normalize(feats, dim=1)


class Adam:
    """torch.optim.Adam on stacked parameters, with one learning rate per run."""

    def __init__(self, P, lr, weight_decay, betas=(0.9, 0.999), eps=1e-8):
        self.P, self.betas, self.eps, self.t = P, betas, eps, 0
        self.lr, self.wd = lr, weight_decay
        self.has_wd = float(weight_decay.abs().max()) > 0
        self.m = {k: torch.zeros_like(v) for k, v in P.items()}
        self.v = {k: torch.zeros_like(v) for k, v in P.items()}

    def reset(self):
        self.t = 0
        for k in self.P:
            self.m[k].zero_()
            self.v[k].zero_()

    @torch.no_grad()
    def step(self):
        self.t += 1
        b1, b2 = self.betas
        c1, c2 = 1 - b1 ** self.t, (1 - b2 ** self.t) ** 0.5
        for k, p in self.P.items():
            g = p.grad
            shape = (-1,) + (1,) * (p.dim() - 1)
            if k.startswith('emb'):
                g[:, 0] = 0                    # padding_idx=0 receives no gradient
            if self.has_wd:
                g = g + self.wd.view(shape) * p
            self.m[k].mul_(b1).add_(g, alpha=1 - b1)
            self.v[k].mul_(b2).addcmul_(g, g, value=1 - b2)
            denom = (self.v[k].sqrt() / c2).add_(self.eps)
            p.sub_((self.lr / c1).view(shape) * self.m[k] / denom)
            p.grad = None


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------
class Job:
    def __init__(self, algorithm, env, seed, steps, checkpoint_freq, hparams, out, data_dir,
                 stream='rounds', n_tasks=4):
        self.algorithm, self.env, self.seed, self.out = algorithm, env, seed, Path(out)
        self.stream, self.n_tasks = stream, n_tasks
        self.test_envs = [env] if env >= 0 else []
        self.hparams_json = hparams
        self.hparams = hparams_registry.default_hparams(algorithm)
        if hparams:
            self.hparams.update(json.loads(hparams))
        self.steps_arg, self.freq_arg, self.data_dir = steps, checkpoint_freq, data_dir
        self.n_steps = steps or datasets.AbRosetta.N_STEPS
        self.freq = checkpoint_freq or datasets.AbRosetta.CHECKPOINT_FREQ
        self.hparams['n_steps'] = self.n_steps
        self.train_envs = [e for e in range(n_tasks) if e not in self.test_envs]
        cls = algorithms.get_algorithm_class(algorithm)
        self.variant = algorithm if algorithm in ERM_FAMILY else ''
        if algorithm in ERM_FAMILY:
            self.kind = 'erm'
        else:
            self.kind = 'replay' if cls.REPLAY else 'finetune'
            self.dual_weights, self.adaptive = cls.DUAL_WEIGHTS, cls.ADAPTIVE
            self.farthest = cls.SELECTION == 'farthest'
            # hparam selection='dual': store examples in proportion to their
            # sample-level dual variables (Elenter et al., Sec. 4). Off by default.
            self.dual_select = self.hparams.get('selection') == 'dual'
            self.domain_steps = self.hparams['domain_steps']
            if self.domain_steps <= 0:
                self.domain_steps = max(1, self.n_steps // len(self.train_envs))

    @property
    def group(self):
        return (self.kind, len(self.train_envs), self.n_steps, self.freq,
                self.hparams['batch_size'], getattr(self, 'domain_steps', 0), self.variant,
                self.stream)

    def args_record(self, code_hash):
        return {
            'algorithm': self.algorithm, 'batched': True, 'checkpoint_freq': self.freq_arg,
            'code_hash': code_hash, 'data_dir': self.data_dir, 'dataset': 'AbRosetta',
            'holdout_fraction': HOLDOUT_FRACTION, 'hparams': self.hparams_json,
            'hparams_seed': 0, 'output_dir': str(self.out), 'seed': self.seed,
            'steps': self.steps_arg, 'stream': self.stream, 'task': 'domain_generalization',
            'test_envs': self.test_envs, 'trial_seed': self.seed, 'use_esm': False,
        }

    def initial_weights(self, input_shape):
        """The weights train.py would start this run from."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        cls = algorithms.get_algorithm_class(self.algorithm)
        with contextlib.redirect_stdout(io.StringIO()):
            algo = cls(input_shape, 2, len(self.train_envs), dict(self.hparams))
        f, c = algo.network[0], algo.network[1]
        return {'emb_ab': f.emb_ab.weight, 'emb_ag': f.emb_ag.weight,
                'c1w': f.conv_layers[0].weight, 'c1b': f.conv_layers[0].bias,
                'c2w': f.conv_layers[2].weight, 'c2b': f.conv_layers[2].bias,
                'fw': f.ffd.weight, 'fb': f.ffd.bias, 'cw': c.weight, 'cb': c.bias}


def partition_sizes(duals, adaptive, alpha, buffer_size):
    """Eq. 8: n_k = |B| (alpha * lambda_k / ||lambda||_1 + (1 - alpha) / t)."""
    t = duals.numel()
    uniform = torch.full((t,), 1.0 / t)
    total = duals.sum()
    if adaptive and total > 1e-8:
        share = alpha * duals / total + (1 - alpha) * uniform
    else:
        share = uniform
    return [int(s) for s in (buffer_size * share).floor()]


# --------------------------------------------------------------------------
# one group of runs that share a schedule
# --------------------------------------------------------------------------
def run_group(jobs, data, device, code_hash):
    kind, D, n_steps, freq, bs, domain_steps, variant, _ = jobs[0].group
    T = len(data['sizes'])                              # tasks in the stream
    M, N = len(jobs), data['tok'].size(0)
    sizes, offsets = data['sizes'], data['offsets']
    ar = torch.arange(M, device=device)

    # ---- splits ----------------------------------------------------------
    split_cache = {}
    def split(seed, e):
        if (seed, e) not in split_cache:
            split_cache[seed, e] = split_keys(sizes[e], seed, e)
        return split_cache[seed, e]

    in_len = torch.zeros(M, D, dtype=torch.long)
    in_tab = torch.zeros(M, D, max(sizes), dtype=torch.long)
    part = torch.zeros(M, N, dtype=torch.long)          # 2 * env + (0 in, 1 out)
    for j, job in enumerate(jobs):
        for e in range(T):
            keys_in, keys_out = split(job.seed, e)
            part[j, offsets[e] + np.array(keys_in)] = 2 * e
            part[j, offsets[e] + np.array(keys_out)] = 2 * e + 1
        for d, e in enumerate(job.train_envs):
            keys_in = split(job.seed, e)[0]
            in_len[j, d] = len(keys_in)
            in_tab[j, d, :len(keys_in)] = torch.as_tensor(keys_in) + int(offsets[e])
    in_tab, part = in_tab.to(device), part.to(device)
    part_count = torch.zeros(M, 2 * T, device=device).scatter_add_(
        1, part, torch.ones(M, N, device=device))
    steps_per_epoch = min(len(split(jobs[0].seed, e)[0]) for e in range(T)) / bs
    label = data['y'].argmax(1)

    # ---- weights, optimizer, random streams ------------------------------
    init = [job.initial_weights(data['input_shape']) for job in jobs]
    P = {k: torch.stack([w[k].detach() for w in init]).to(device).requires_grad_()
         for k in PARAMS}
    lr_key = 'lr' if kind == 'erm' else 'primal_lr'
    opt = Adam(P, torch.tensor([job.hparams[lr_key] for job in jobs], device=device),
               torch.tensor([job.hparams['weight_decay'] for job in jobs],
                            dtype=torch.float32, device=device))
    batch_gen = [torch.Generator(device=device).manual_seed(job.seed) for job in jobs]
    np_rng = [np.random.RandomState(job.seed) for job in jobs]
    group_seed = int(hashlib.md5(','.join(str(j.out) for j in jobs).encode()).hexdigest(), 16)
    gen = torch.Generator(device=device).manual_seed(group_seed % (2 ** 31))

    # ---- state of the sequential methods ---------------------------------
    sequential = kind != 'erm'
    if sequential:
        cap = max(bs, max(job.hparams['buffer_size'] for job in jobs))
        dual = torch.zeros(M, D, device=device)
        eps = torch.zeros(M, D, device=device)
        mem_idx = torch.zeros(M, D, cap, dtype=torch.long, device=device)
        mem_n = torch.zeros(M, D, dtype=torch.long, device=device)
        ring = torch.zeros(M, RECENT, device=device)
        ring_count = 0
        pools = [np.zeros(job.hparams['pool_size'], dtype=np.int64) for job in jobs]
        pool_n, seen = [0] * M, [0] * M
        dual_lr = torch.tensor([job.hparams['dual_lr'] for job in jobs], device=device).view(M, 1)
        use_dual = torch.tensor([getattr(job, 'dual_weights', False) for job in jobs],
                                device=device).view(M, 1)
        replay_weight = torch.tensor([job.hparams.get('replay_weight', 1.0) for job in jobs],
                                     device=device).view(M, 1)
        # Stored examples replayed per earlier task per step (hparam replay_batch;
        # absent or 0 = batch_size, the behaviour of every run before F2).
        replay_batch = torch.tensor([job.hparams.get('replay_batch', 0) or bs for job in jobs],
                                    device=device).clamp(max=bs).view(M, 1, 1)
        # Sample-level dual variables (hparam selection='dual'), one per example.
        dual_select = torch.tensor([job.dual_select for job in jobs], device=device).view(M, 1)
        any_dual_select = bool(dual_select.any())
        if any_dual_select:
            ex_dual = torch.zeros(M, N, device=device)
            eps_scale = torch.tensor([job.hparams['epsilon_scale'] for job in jobs], device=device)
            eps_fixed = torch.tensor([job.hparams['epsilon'] for job in jobs], device=device)

    def sample_dual_step(rows, ce, level, valid=None):
        """lambda_i <- [lambda_i + eta_d (loss_i - epsilon_i)]_+ on the given examples."""
        delta = dual_lr * (ce - level) * dual_select
        if valid is not None:
            delta = delta * valid
        ex_dual.scatter_add_(1, rows, delta)
        ex_dual.scatter_(1, rows, ex_dual.gather(1, rows).clamp(min=0))

    def selection_record(j, t, pool, keep):
        """What the dual-selected examples of task t look like against their pool."""
        Pj = {k: v[j:j + 1] for k, v in P.items()}
        logits = torch.cat([forward(Pj, data, c.unsqueeze(0))[1][0] for c in pool.split(256)])
        ce = -(F.log_softmax(logits, dim=1) * data['y'][pool]).sum(1)
        lam = ex_dual[j, pool]
        chosen = torch.isin(pool, keep)
        mean = lambda v, m: float(v[m].float().mean()) if bool(m.any()) else float('nan')
        every = torch.ones_like(chosen)
        r = {'task': t, 'pool_n': pool.numel(), 'kept_n': keep.numel(),
             'dual_nonzero': mean(lam > 0, every), 'dual_mean': mean(lam, every),
             'dual_max': float(lam.max()), 'dual_std': float(lam.std()),
             'kept_dual_nonzero': mean(lam > 0, chosen)}
        for name, m in (('pool', every), ('kept', chosen)):
            r[f'{name}_loss'] = mean(ce, m)
            r[f'{name}_binder'] = mean(label[pool], m)
            r[f'{name}_wrong'] = mean(logits.argmax(1) != label[pool], m)
        with open(REPO_ROOT / jobs[j].out / 'selection.jsonl', 'a') as f:
            f.write(json.dumps(r, sort_keys=True) + '\n')
    t = 0                                              # current phase

    def select(j, rows, n):
        """n of the given examples: at random, or farthest-first in feature space."""
        size = rows.numel()
        n = min(n, size)
        if n == 0:
            return rows[:0]
        if jobs[j].dual_select and n < size:
            # Fill Buffer of the paper: draw without replacement with probability
            # proportional to the sample duals (zero duals only once the rest ran out).
            w = ex_dual[j, rows].double().cpu().numpy() + 1e-8
            return rows[torch.as_tensor(np_rng[j].choice(size, n, replace=False, p=w / w.sum()),
                                        device=device)]
        if not jobs[j].farthest or n == size:
            return rows[torch.as_tensor(np_rng[j].permutation(size)[:n], device=device)]
        feats = features_of(P, j, data, rows)
        chosen = [int(np_rng[j].randint(size))]
        min_dist = 1 - feats @ feats[chosen[0]]
        for _ in range(n - 1):
            min_dist[chosen[-1]] = -1
            chosen.append(int(min_dist.argmax()))
            min_dist = torch.minimum(min_dist, 1 - feats @ feats[chosen[-1]])
        return rows[torch.tensor(chosen, device=device)]

    def end_of_domain(t):
        """Phase t is over: set its constraint level and refill each memory."""
        recent = ring[:, :min(ring_count, RECENT)].double().mean(1).cpu()
        duals_cpu = dual.detach().cpu().float()
        for j, job in enumerate(jobs):
            scale = job.hparams['epsilon_scale']
            eps[j, t] = scale * float(recent[j]) if scale > 0 else job.hparams['epsilon']
            duals = duals_cpu[j, :t + 1].clone()
            if t > 0:
                duals[t] = duals[:t].mean()
            share = partition_sizes(duals, job.adaptive, job.hparams['alpha'],
                                    job.hparams['buffer_size'])
            used = 0
            for k in range(t):
                n_k = int(mem_n[j, k])
                keep = select(j, mem_idx[j, k, :n_k], min(share[k], n_k))
                mem_idx[j, k, :keep.numel()] = keep
                mem_n[j, k] = keep.numel()
                used += keep.numel()
            pool = torch.as_tensor(pools[j][:pool_n[j]], device=device)
            keep = select(j, pool, job.hparams['buffer_size'] - used)
            mem_idx[j, t, :keep.numel()] = keep
            mem_n[j, t] = keep.numel()
            if job.dual_select:
                selection_record(j, t, pool, keep)
            pool_n[j], seen[j] = 0, 0

    # ---- logging ---------------------------------------------------------
    acc_keys = ['loss'] + (['current_loss'] + [f'replay_loss_{k}' for k in range(D - 1)]
                           if sequential else [])
    if variant in ('IRM', 'VREx', 'CORAL', 'SD'):
        acc_keys.append('penalty')

    # ---- the benchmark's joint-training methods ---------------------------
    def hp(name):
        return torch.tensor([float(job.hparams[name]) for job in jobs], device=device)
    if variant in ('IRM', 'VREx'):
        prefix = variant.lower()
        strength, anneal = hp(f'{prefix}_lambda'), jobs[0].hparams[f'{prefix}_penalty_anneal_iters']
        assert all(job.hparams[f'{prefix}_penalty_anneal_iters'] == anneal for job in jobs)
    elif variant == 'GroupDRO':
        eta, q = hp('groupdro_eta').view(M, 1), torch.ones(M, D, device=device)
    elif variant == 'CORAL':
        strength = hp('mmd_gamma')
    elif variant == 'SD':
        strength = hp('sd_reg')

    def joint_loss(step, feats, logits, y, ce):
        """(objective, logged loss, penalty) of each run; ce: [M, D * bs]."""
        env_loss = ce.view(M, D, bs).mean(2)
        if variant == 'ERM':
            return ce.mean(1), ce.mean(1), None
        if variant == 'SD':
            penalty = (logits ** 2).mean((1, 2))
            return ce.mean(1) + strength * penalty, ce.mean(1), penalty
        if variant == 'GroupDRO':
            with torch.no_grad():
                q.mul_((eta * env_loss).exp())
                q.div_(q.sum(1, keepdim=True))
            loss = (env_loss * q).sum(1)
            return loss, loss, None
        if variant == 'CORAL':
            f = feats.view(M, D, bs, -1)
            mean = f.mean(2)
            cent = f - mean.unsqueeze(2)
            cova = cent.transpose(2, 3) @ cent / (bs - 1)
            penalty = 0
            for a, b in itertools.combinations(range(D), 2):
                penalty = penalty + (mean[:, a] - mean[:, b]).pow(2).mean(1) \
                    + (cova[:, a] - cova[:, b]).pow(2).mean((1, 2))
            penalty = penalty / (D * (D - 1) / 2)
            return env_loss.mean(1) + strength * penalty, env_loss.mean(1), penalty
        # IRM and V-REx: weight 1 until the anneal step, then the full penalty,
        # with the optimizer state reset at the switch.
        if step == anneal:
            opt.reset()
        weight = strength if step >= anneal else torch.ones_like(strength)
        if variant == 'VREx':
            mean = env_loss.mean(1)
            penalty = (env_loss - mean.unsqueeze(1)).pow(2).mean(1)
        else:
            # d/ds of the cross-entropy of (s * logits) at s = 1, on the even
            # and on the odd examples of each environment's batch.
            g = ((F.softmax(logits, dim=2) - y) * logits).sum(2).view(M, D, bs)
            penalty = (g[:, :, 0::2].mean(2) * g[:, :, 1::2].mean(2)).mean(1)
            mean = env_loss.mean(1)
        loss = mean + weight * penalty
        return loss, loss, penalty
    sums = {k: torch.zeros(M, device=device) for k in acc_keys}
    n_acc, t_acc = 0, time.time()
    for job in jobs:
        (REPO_ROOT / job.out).mkdir(parents=True, exist_ok=True)
        (REPO_ROOT / job.out / 'results.jsonl').unlink(missing_ok=True)
        (REPO_ROOT / job.out / 'selection.jsonl').unlink(missing_ok=True)

    def checkpoint(step):
        nonlocal n_acc, t_acc
        torch.cuda.synchronize()
        step_time = (time.time() - t_acc) / max(n_acc, 1)
        logits = forward_all(P, data)
        correct = (logits.argmax(2) == label).float()
        ce = -(F.log_softmax(logits, dim=2) * data['y']).sum(2)
        acc = (torch.zeros(M, 2 * T, device=device).scatter_add_(1, part, correct)
               / part_count).cpu().numpy()
        loss = (torch.zeros(M, 2 * T, device=device).scatter_add_(1, part, ce)
                / part_count).cpu().numpy()
        n_correct = torch.zeros(M, 2 * T, device=device).scatter_add_(1, part, correct).cpu().numpy()
        counts = part_count.cpu().numpy()
        means = {k: (v / max(n_acc, 1)).cpu().numpy() for k, v in sums.items()}
        if sequential:
            state = {'dual': dual.cpu().numpy(), 'eps': eps.cpu().numpy(),
                     'buf': mem_n.cpu().numpy()}
            # Loss and accuracy on all stored examples of each task (not a sampled batch).
            stored = (torch.arange(cap, device=device) < mem_n[:, :, None]).float()
            at_mem = lambda v: ((v.gather(1, mem_idx.flatten(1)).view(M, D, cap) * stored).sum(2)
                                / stored.sum(2)).cpu().numpy()       # nan for an empty memory
            mem_loss, mem_acc = at_mem(ce), at_mem(correct)
        mem_gb = torch.cuda.max_memory_allocated() / 2 ** 30
        for j, job in enumerate(jobs):
            r = {'step': step, 'epoch': step / steps_per_epoch, 'step_time': step_time,
                 'mem_gb': mem_gb}
            for k in acc_keys:
                r[k] = float(means[k][j])
            if sequential:
                for k in range(D - 1):
                    r[f'dual_{k}'] = float(state['dual'][j, k])
                    r[f'eps_{k}'] = float(state['eps'][j, k])
                    r[f'buf_{k}'] = float(state['buf'][j, k])
                    r[f'mem_loss_{k}'] = float(mem_loss[j, k])
                    r[f'mem_acc_{k}'] = float(mem_acc[j, k])
            for e in range(T):
                r[f'env{e}_in_acc'] = float(acc[j, 2 * e])
                r[f'env{e}_out_acc'] = float(acc[j, 2 * e + 1])
                r[f'env{e}_in_loss'] = float(loss[j, 2 * e])
                r[f'env{e}_out_loss'] = float(loss[j, 2 * e + 1])
            val = [2 * e + 1 for e in job.train_envs]
            r['agg_val_acc'] = float(acc[j, val].sum() / (len(val) + 1e-9))
            r['agg_val_loss'] = float(loss[j, val].mean())
            test = [2 * e + s for e in job.test_envs for s in (0, 1)]
            r['agg_test_acc'] = (float(n_correct[j, test].sum() / counts[j, test].sum())
                                 if test else float('nan'))
            r['hparams'] = job.hparams
            r['args'] = job.args_record(code_hash)
            with open(REPO_ROOT / job.out / 'results.jsonl', 'a') as f:
                f.write(json.dumps(r, sort_keys=True) + '\n')
        for v in sums.values():
            v.zero_()
        n_acc, t_acc = 0, time.time()

    # ---- training --------------------------------------------------------
    y1h = data['y']
    for step in range(n_steps):
        # The minibatch stream of train.py: one index batch per training env,
        # drawn in env order from the run's own generator.
        idx = torch.stack([torch.stack([
            torch.randint(int(in_len[j, d]), (bs,), device=device, generator=batch_gen[j])
            for d in range(D)]) for j in range(M)])                      # [M, D, bs]
        rows_env = in_tab.gather(2, idx)                                 # [M, D, bs]

        if not sequential:
            rows = rows_env.flatten(1)
            feats, logits = forward(P, data, rows)
            ce = -(F.log_softmax(logits, dim=2) * y1h[rows]).sum(2)
            loss, shown, penalty = joint_loss(step, feats, logits, y1h[rows], ce)
        else:
            last = t + 1 == D
            cur_rows = rows_env[:, t]
            if kind == 'replay' and not last:
                # Reservoir sampling of the phase's stream into the staging pool.
                cur_cpu = cur_rows.cpu().numpy()
                for j in range(M):
                    size = pools[j].size
                    room = size - pool_n[j]
                    if room > 0:
                        take = min(room, bs)
                        pools[j][pool_n[j]:pool_n[j] + take] = cur_cpu[j, :take]
                        pool_n[j] += take
                    i = np.arange(max(room, 0), bs)
                    if i.size:
                        slot = np_rng[j].randint(0, seen[j] + i + 1)
                        hit = slot < size
                        pools[j][slot[hit]] = cur_cpu[j, i[hit]]
                    seen[j] += bs
            n_rep = t if kind == 'replay' else 0
            rows = cur_rows
            if n_rep:
                # A random batch from each earlier domain's memory.
                score = torch.rand(M, n_rep, cap, device=device, generator=gen)
                slots = torch.arange(cap, device=device)
                score[slots >= mem_n[:, :n_rep, None]] = 2.0
                pick = score.topk(bs, dim=2, largest=False).indices
                # topk returns the lowest scores in order, so its first
                # replay_batch entries are a uniform draw without replacement.
                valid = (torch.arange(bs, device=device)
                         < torch.minimum(mem_n[:, :n_rep, None], replay_batch)).float()
                rep_rows = mem_idx[:, :n_rep].gather(2, pick)
                rows = torch.cat([cur_rows, rep_rows.flatten(1)], dim=1)
            ce = -(F.log_softmax(forward(P, data, rows)[1], dim=2) * y1h[rows]).sum(2)
            cur = ce[:, :bs].mean(1)
            loss = cur
            if n_rep:
                has = (mem_n[:, :n_rep] > 0).float()
                rep = (ce[:, bs:].view(M, n_rep, bs) * valid).sum(2) / valid.sum(2).clamp(min=1)
                weight = torch.where(use_dual, dual[:, :n_rep], replay_weight.expand(M, n_rep))
                loss = cur + (has * weight * (rep - eps[:, :n_rep])).sum(1)

        loss.sum().backward()
        opt.step()

        with torch.no_grad():
            sums['loss'] += (loss if sequential else shown).detach()
            if not sequential and penalty is not None:
                sums['penalty'] += penalty.detach()
            if sequential:
                sums['current_loss'] += cur.detach()
                ring[:, ring_count % RECENT] = cur.detach()
                ring_count += 1
                if n_rep:
                    rep = rep.detach()
                    # Dual ascent: lambda_k <- [lambda_k + eta_d (loss_k - epsilon_k)]_+
                    stepped = torch.clamp(dual[:, :n_rep] + dual_lr * (rep - eps[:, :n_rep]), min=0)
                    dual[:, :n_rep] = torch.where(use_dual & (has > 0), stepped, dual[:, :n_rep])
                    for k in range(n_rep):
                        sums[f'replay_loss_{k}'] += rep[:, k] * has[:, k]
                if any_dual_select and kind == 'replay':
                    ce = ce.detach()
                    if not last:
                        # Current task: its constraint level is not fixed until the
                        # phase ends, so use the level it would get now.
                        recent = ring[:, :min(ring_count, RECENT)].mean(1)
                        level = torch.where(eps_scale > 0, eps_scale * recent, eps_fixed)
                        sample_dual_step(cur_rows, ce[:, :bs], level.view(M, 1))
                    if n_rep:
                        sample_dual_step(rows[:, bs:], ce[:, bs:],
                                         eps[:, :n_rep].repeat_interleave(bs, dim=1),
                                         valid.expand(M, n_rep, bs).flatten(1))
                if (step + 1) % domain_steps == 0 and not last:
                    if kind == 'replay':
                        end_of_domain(t)
                    t += 1
                    ring_count = 0
            n_acc += 1
            if step % freq == 0 or step == n_steps - 1:
                checkpoint(step)

    for job in jobs:
        (REPO_ROOT / job.out / 'done').write_text('done')


# --------------------------------------------------------------------------
def build_jobs(spec, data_dir):
    jobs = []
    stream = spec.get('stream') or 'rounds'
    n_tasks = 4 if stream == 'rounds' else len(streams.task_order(data_dir, stream)[1])
    envs = spec.get('envs', [0, 1, 2, 3])
    for alg, env, seed in itertools.product(spec['algorithms'], envs, spec.get('seeds', [0, 1, 2])):
        tag = spec.get('tag')
        out = Path(spec.get('output_root', 'outputs/sweep')) / (alg + (f"__{tag}" if tag else "")) \
            / f"env{env}_seed{seed}"
        if (REPO_ROOT / out / 'done').exists():
            continue
        hp = spec.get('hparams')
        jobs.append(Job(alg, env, seed, spec.get('steps'), spec.get('checkpoint_freq'),
                        json.dumps(hp) if isinstance(hp, dict) else hp, out, data_dir,
                        stream, n_tasks))
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--spec', help='JSON file with a list of sweeps (same fields as the flags)')
    p.add_argument('--algorithms', nargs='+', default=['ERM', 'Finetune', 'ER', 'ER_FF', 'PDCL_RS', 'PDCL'])
    p.add_argument('--envs', nargs='+', type=int, default=[0, 1, 2, 3])
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--output_root', default='outputs/sweep')
    p.add_argument('--data_dir', default='data')
    p.add_argument('--steps', type=int, default=None)
    p.add_argument('--checkpoint_freq', type=int, default=None)
    p.add_argument('--hparams', default=None)
    p.add_argument('--tag', default=None)
    p.add_argument('--stream', default='rounds',
                   help='how the examples are grouped into tasks (see domainbed/streams.py)')
    p.add_argument('--max_models', type=int, default=96, help='runs trained together')
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()

    if args.spec:
        specs = json.loads(Path(args.spec).read_text())
    else:
        specs = [{k: getattr(args, k) for k in (
            'algorithms', 'envs', 'seeds', 'output_root', 'steps', 'checkpoint_freq',
            'hparams', 'tag', 'stream')}]
    jobs = [job for spec in specs for job in build_jobs(spec, args.data_dir)]
    groups = {}
    for job in jobs:
        groups.setdefault(job.group, []).append(job)
    chunks = [g[i:i + args.max_models] for g in groups.values()
              for i in range(0, len(g), args.max_models)]
    print(f"{len(jobs)} runs in {len(chunks)} batches", flush=True)
    if args.dry_run or not jobs:
        for chunk in chunks:
            print(chunk[0].group, len(chunk))
        return

    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    device = 'cuda'
    root = Path(__file__).resolve().parents[1]
    code_files = sorted(f for f in root.rglob('*.py')
                        if f.parent != root / 'scripts' or f.name in ('train.py', 'train_batched.py'))
    code_hash = hashlib.sha256(b''.join(f.read_bytes() for f in code_files)).hexdigest()[:12]
    tokens, data, loaded = load_tokens(args.data_dir, device), None, None
    for i, chunk in enumerate(chunks):
        if chunk[0].stream != loaded:
            loaded = chunk[0].stream
            data = load_data(args.data_dir, device, loaded, tokens)
        start = time.time()
        print(f"batch {i + 1}/{len(chunks)}: {len(chunk)} runs {chunk[0].group}", flush=True)
        run_group(chunk, data, device, code_hash)
        print(f"  finished in {time.time() - start:.0f}s", flush=True)
    print("all jobs finished")


if __name__ == '__main__':
    main()
