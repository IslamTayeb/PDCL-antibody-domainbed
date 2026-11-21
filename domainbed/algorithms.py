"""Training algorithms.

Joint methods (ERM and the domain generalization methods) see a minibatch of
every training round at each step. Sequential methods (ER and its subclasses)
see the rounds one after another.
"""
import collections

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from domainbed import networks

ALGORITHMS = ['ERM', 'IRM', 'VREx', 'GroupDRO', 'CORAL', 'SD', 'Finetune', 'ER', 'ER_FF', 'PDCL_RS', 'PDCL']


def get_algorithm_class(algorithm_name):
    if algorithm_name not in ALGORITHMS:
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(nn.Module):
    """update(minibatches) takes one training step on a list of (x, y) pairs, one per training round."""

    def __init__(self, input_shape, num_classes, num_domains, hparams, lr_key='lr'):
        super().__init__()
        self.hparams = hparams
        self.lr_key = lr_key
        self.featurizer = networks.Featurizer(input_shape)
        self.classifier = networks.Classifier(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = self.new_optimizer()

    def new_optimizer(self):
        return torch.optim.Adam(self.network.parameters(), lr=self.hparams[self.lr_key],
                                weight_decay=self.hparams['weight_decay'])

    def predict(self, x):
        return self.network(x)

    def step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ER(Algorithm):
    """
    Experience Replay over training domains presented sequentially
    (Chaudhry et al., 2019), and the base class for the primal-dual variants.

    Training is split into `num_domains` phases of `domain_steps` steps. In
    phase t the model sees minibatches of domain t only, plus replay
    minibatches from a memory holding at most `buffer_size` examples of
    domains 0..t-1 in total.

    Memory is strict: once an example is dropped it is gone, so the share of
    an earlier domain can only shrink. While domain t is current, a staging
    pool of up to `pool_size` of its examples is kept by reservoir sampling;
    at the end of the phase the memory is re-partitioned, earlier domains are
    cut down to their new share, domain t's share is selected from its pool,
    and the pool is discarded.

    Subclasses switch on independent components:
      REPLAY        use the memory at all (else plain sequential fine-tuning)
      DUAL_WEIGHTS  replay losses weighted by dual variables (else weight 1)
      ADAPTIVE      memory partitioned by dual variables (else uniform)
      SELECTION     'random' or 'farthest' (farthest-first in feature space)
    """
    REPLAY = True
    DUAL_WEIGHTS = False
    ADAPTIVE = False
    SELECTION = 'random'

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams, lr_key='primal_lr')
        self.num_domains = num_domains
        self.buffer_size = hparams['buffer_size']
        self.pool_size = hparams['pool_size']
        self.alpha = hparams['alpha']
        self.dual_lr = hparams['dual_lr']
        self.domain_steps = hparams['domain_steps']
        if self.domain_steps <= 0:
            self.domain_steps = max(1, hparams['n_steps'] // num_domains)

        self.register_buffer('update_count', torch.tensor(0))
        self.register_buffer('dual_vars', torch.zeros(num_domains))
        self.register_buffer('epsilon', torch.zeros(num_domains))
        self.current_domain = 0
        self.pool = None                          # staging pool of the current domain
        self.seen = 0                             # examples offered to the pool
        self.memory = {}                          # domain -> (x, y)
        self.recent_losses = collections.deque(maxlen=200)

    def _add_to_pool(self, x, y):
        """Reservoir sampling: the pool is a uniform sample of the stream."""
        if self.pool is None:
            self.pool, self.seen = (x[:0].clone(), y[:0].clone()), 0
        px, py = self.pool
        room = self.pool_size - px.size(0)
        if room > 0:
            px = torch.cat([px, x[:room]])
            py = torch.cat([py, y[:room]])
        for i in range(max(room, 0), x.size(0)):
            j = np.random.randint(0, self.seen + i + 1)
            if j < self.pool_size:
                px[j], py[j] = x[i], y[i]
        self.pool, self.seen = (px, py), self.seen + x.size(0)

    def _features(self, x):
        was_training = self.network.training
        self.network.eval()
        with torch.no_grad():
            feats = torch.cat([self.featurizer(chunk) for chunk in x.split(256)])
        self.network.train(was_training)
        return F.normalize(feats, dim=1)

    def _select(self, x, n):
        """Indices of n of the examples in x."""
        size = x.size(0)
        n = min(n, size)
        if n == 0:
            return torch.empty(0, dtype=torch.long, device=x.device)
        if self.SELECTION == 'random' or n == size:
            return torch.randperm(size, device=x.device)[:n]
        # Farthest-first traversal (greedy k-center) under cosine distance.
        feats = self._features(x)
        chosen = [np.random.randint(size)]
        min_dist = 1 - feats @ feats[chosen[0]]
        for _ in range(n - 1):
            min_dist[chosen[-1]] = -1
            chosen.append(int(min_dist.argmax()))
            min_dist = torch.minimum(min_dist, 1 - feats @ feats[chosen[-1]])
        return torch.tensor(chosen, device=x.device)

    def _partition_sizes(self, duals):
        """n_k = |B| (alpha * lambda_k / ||lambda||_1 + (1 - alpha) / t)."""
        t = duals.numel()
        uniform = torch.full((t,), 1.0 / t)
        total = duals.sum()
        if self.ADAPTIVE and total > 1e-8:
            share = self.alpha * duals / total + (1 - self.alpha) * uniform
        else:
            share = uniform
        return [int(s) for s in (self.buffer_size * share).floor()]

    def _end_of_domain(self, t):
        """Domain t is finished: set its constraint level and refill the memory."""
        scale = self.hparams['epsilon_scale']
        if scale > 0:
            # Constraint level relative to the loss reached on the task
            # (Elenter et al., 2023, App. A.10 use about 1.1x).
            self.epsilon[t] = scale * float(np.mean(self.recent_losses))
        else:
            self.epsilon[t] = self.hparams['epsilon']
        self.recent_losses.clear()

        # Domain t was never a constraint, so it has no dual variable; it
        # enters the partition with the mean dual of the earlier domains.
        duals = self.dual_vars[:t + 1].detach().cpu().float().clone()
        if t > 0:
            duals[t] = duals[:t].mean()
        sizes = self._partition_sizes(duals)

        used = 0
        for k in range(t):
            mx, my = self.memory[k]
            keep = self._select(mx, min(sizes[k], mx.size(0)))
            self.memory[k] = (mx[keep], my[keep])
            used += keep.numel()
        px, py = self.pool
        keep = self._select(px, self.buffer_size - used)
        self.memory[t] = (px[keep].clone(), py[keep].clone())
        self.pool = None
        self.current_domain = t + 1

    def update(self, minibatches):
        t = self.current_domain
        last_domain = t + 1 == self.num_domains
        x, y = minibatches[t]
        if self.REPLAY and not last_domain:
            self._add_to_pool(x, y)

        loss = F.cross_entropy(self.predict(x), y)
        self.recent_losses.append(loss.item())
        replay_losses = {}
        for k in range(t if self.REPLAY else 0):
            mx, my = self.memory[k]
            if mx.size(0) == 0:
                continue
            pick = torch.randperm(mx.size(0), device=mx.device)[:x.size(0)]
            replay_loss = F.cross_entropy(self.predict(mx[pick]), my[pick])
            weight = self.dual_vars[k] if self.DUAL_WEIGHTS else 1.0
            loss = loss + weight * (replay_loss - self.epsilon[k])
            replay_losses[k] = replay_loss.detach()
        self.step(loss)

        if self.DUAL_WEIGHTS:
            # Dual ascent: lambda_k <- [lambda_k + eta_d (loss_k - epsilon_k)]_+
            for k, replay_loss in replay_losses.items():
                self.dual_vars[k] = torch.clamp(
                    self.dual_vars[k] + self.dual_lr * (replay_loss - self.epsilon[k]), min=0.0)

        self.update_count += 1
        if int(self.update_count) % self.domain_steps == 0 and not last_domain:
            if self.REPLAY:
                self._end_of_domain(t)
            else:
                self.current_domain = t + 1

        out = {'loss': loss.item(), 'current_loss': self.recent_losses[-1] if self.recent_losses else 0.0}
        for k in range(self.num_domains - 1):
            out[f'dual_{k}'] = self.dual_vars[k].item()
            out[f'buf_{k}'] = float(self.memory[k][0].size(0) if k in self.memory else 0)
            out[f'eps_{k}'] = self.epsilon[k].item()
            out[f'replay_loss_{k}'] = replay_losses[k].item() if k in replay_losses else 0.0
        return out


class Finetune(ER):
    """Sequential training with no replay."""
    REPLAY = False


class ER_FF(ER):
    """Experience Replay with farthest-first selection."""
    SELECTION = 'farthest'


class PDCL_RS(ER):
    """Primal-dual continual learning with adaptive memory partition and
    random sample selection: the PDCL of Elenter et al. (2023)."""
    DUAL_WEIGHTS = True
    ADAPTIVE = True


class PDCL(PDCL_RS):
    """PDCL_RS with farthest-first sample selection."""
    SELECTION = 'farthest'


class ERM(Algorithm):
    """Empirical risk minimization on all training rounds at once."""

    def update(self, minibatches):
        all_x = torch.cat([x for x, _ in minibatches])
        all_y = torch.cat([y for _, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)
        self.step(loss)
        return {'loss': loss.item()}


class IRM(ERM):
    """Invariant Risk Minimization (Arjovsky et al., 2019)."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        scale = torch.tensor(1., device=logits.device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        return torch.sum(grad_1 * grad_2)

    def update(self, minibatches):
        anneal = self.hparams['irm_penalty_anneal_iters']
        penalty_weight = self.hparams['irm_lambda'] if self.update_count >= anneal else 1.0
        nll = 0.
        penalty = 0.
        all_logits = self.network(torch.cat([x for x, y in minibatches]))
        idx = 0
        for x, y in minibatches:
            logits = all_logits[idx:idx + x.shape[0]]
            idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == anneal:
            # Reset Adam: the gradient magnitude jumps at this step.
            self.optimizer = self.new_optimizer()
        self.step(loss)
        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(), 'penalty': penalty.item()}


class VREx(ERM):
    """V-REx (Krueger et al., 2021)."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches):
        anneal = self.hparams['vrex_penalty_anneal_iters']
        penalty_weight = self.hparams['vrex_lambda'] if self.update_count >= anneal else 1.0
        all_logits = self.network(torch.cat([x for x, y in minibatches]))
        idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[idx:idx + x.shape[0]]
            idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll
        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == anneal:
            self.optimizer = self.new_optimizer()
        self.step(loss)
        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(), 'penalty': penalty.item()}


class GroupDRO(ERM):
    """Group DRO (Sagawa et al., 2020), with training rounds as groups."""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, minibatches):
        device = minibatches[0][0].device
        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)
        losses = torch.zeros(len(minibatches)).to(device)
        for m, (x, y) in enumerate(minibatches):
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()
        self.q /= self.q.sum()
        loss = torch.dot(losses, self.q)
        self.step(loss)
        return {'loss': loss.item()}


class CORAL(ERM):
    """Deep CORAL (Sun & Saenko, 2016): match feature means and covariances across rounds."""

    @staticmethod
    def _distance(x, y):
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)
        return (mean_x - mean_y).pow(2).mean() + (cova_x - cova_y).pow(2).mean()

    def update(self, minibatches):
        objective = 0
        penalty = 0
        nmb = len(minibatches)
        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]
        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self._distance(features[i], features[j])
        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)
        self.step(objective + self.hparams['mmd_gamma'] * penalty)
        if torch.is_tensor(penalty):
            penalty = penalty.item()
        return {'loss': objective.item(), 'penalty': penalty}


class SD(ERM):
    """Spectral decoupling (Pezeshki et al., 2021)."""

    def update(self, minibatches):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_p = self.predict(all_x)
        loss = F.cross_entropy(all_p, all_y)
        penalty = (all_p ** 2).mean()
        self.step(loss + self.hparams["sd_reg"] * penalty)
        return {'loss': loss.item(), 'penalty': penalty.item()}
