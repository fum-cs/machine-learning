"""Minimal discrete Hidden Markov Model utilities for the course notebook.

The class implements the two inference problems that the course covers:

* **Problem 1 - likelihood.** ``P(O | lambda)`` for an observation sequence,
  computed with the forward algorithm and one scaling factor per time step.
* **Problem 2 - decoding.** The most likely state sequence, computed with the
  Viterbi algorithm in log space.

Problem 3 (learning ``A`` and ``B`` with Baum-Welch / EM) is deliberately not
implemented; the notebook only discusses it.

All matrices are row-stochastic::

    start_p[i]    = P(q_1 = i)
    trans_p[i, j] = P(q_t = j | q_{t-1} = i)
    emit_p[i, k]  = P(o_t = observations[k] | q_t = i)

Example
-------
>>> model = hmm(("Rainy", "Sunny"), ("walk", "shop"),
...             [0.6, 0.4],
...             [[0.7, 0.3], [0.4, 0.6]],
...             [[0.1, 0.9], [0.6, 0.4]])
>>> alpha, loglik, scale = model.forward(("walk", "shop", "walk"))
>>> path, logprob = model.viterbi(("walk", "shop", "walk"))
"""

import numpy as np


class hmm:
    """A discrete hidden Markov model with known parameters."""

    def __init__(self, states, observations, start_prob, trans_prob, emit_prob):
        self.states = tuple(states)
        self.observations = tuple(observations)

        self.start_p = np.asarray(start_prob, dtype=float).reshape(-1)
        self.trans_p = np.asarray(trans_prob, dtype=float)
        self.emit_p = np.asarray(emit_prob, dtype=float)

        n, m = len(self.states), len(self.observations)
        expected = {
            "start": (n,),
            "transition": (n, n),
            "emission": (n, m),
        }
        found = {
            "start": self.start_p.shape,
            "transition": self.trans_p.shape,
            "emission": self.emit_p.shape,
        }
        for name, shape in expected.items():
            if found[name] != shape:
                raise ValueError(
                    f"{name} matrix has shape {found[name]}, expected {shape} "
                    f"for {n} states and {m} observations"
                )
        self._check_rows()

        self._obs_index = {o: k for k, o in enumerate(self.observations)}

    def __repr__(self):
        return f"hmm(states={self.states}, observations={self.observations})"

    # ------------------------------------------------------------------ #
    # validation helpers
    # ------------------------------------------------------------------ #
    def _check_rows(self):
        """Reject matrices whose rows are not probability distributions."""
        checks = {
            "start": self.start_p.reshape(1, -1),
            "transition": self.trans_p,
            "emission": self.emit_p,
        }
        for name, matrix in checks.items():
            sums = matrix.sum(axis=1)
            if np.any(np.abs(sums - 1.0) > 1e-6):
                raise ValueError(
                    f"rows of the {name} matrix must sum to 1; got {np.round(sums, 6)}"
                )

    def _to_indices(self, obs):
        try:
            return [self._obs_index[o] for o in obs]
        except KeyError as exc:
            raise ValueError(
                f"unknown observation {exc.args[0]!r}; "
                f"expected one of {self.observations}"
            ) from None

    # ------------------------------------------------------------------ #
    # problem 1: likelihood
    # ------------------------------------------------------------------ #
    def forward(self, obs):
        """Run the forward algorithm.

        Returns
        -------
        alpha : ndarray, shape (T, N)
            ``alpha[t, j]`` is the *scaled* forward variable of state ``j`` at
            time ``t``; every row sums to 1.
        loglik : float
            ``log P(O | lambda)`` accumulated from the scaling factors.
        scale : ndarray, shape (T,)
            The scaling factor used at every time step.
        """
        idx = self._to_indices(obs)
        t_max, n_states = len(idx), len(self.states)
        if t_max == 0:
            raise ValueError("the observation sequence is empty")

        alpha = np.zeros((t_max, n_states))
        scale = np.zeros(t_max)

        alpha[0] = self.start_p * self.emit_p[:, idx[0]]
        scale[0] = alpha[0].sum()
        if scale[0] == 0.0:
            raise ValueError("the first observation has zero probability in every state")
        alpha[0] /= scale[0]

        for t in range(1, t_max):
            alpha[t] = (alpha[t - 1] @ self.trans_p) * self.emit_p[:, idx[t]]
            scale[t] = alpha[t].sum()
            if scale[t] == 0.0:
                raise ValueError(
                    f"observation {obs[t]!r} at position {t + 1} "
                    "has zero probability in every state"
                )
            alpha[t] /= scale[t]

        loglik = float(np.log(scale).sum())
        return alpha, loglik, scale

    def forward_algo(self, obs, digits=6):
        """Print the forward table and return ``(alpha, loglik)``."""
        alpha, loglik, _ = self.forward(obs)

        print("Forward algorithm (scaled alpha; each row sums to 1)")
        print("t   " + "".join(f"{s:>12}" for s in self.states))
        for t, row in enumerate(alpha):
            cells = "".join(f"{v:>12.{digits}g}" for v in row)
            print(f"{t + 1:<4}" + cells)
        print(f"\nP(O | lambda) = {np.exp(loglik):.6g}    log P = {loglik:.6f}")
        return alpha, loglik

    # ------------------------------------------------------------------ #
    # problem 2: decoding
    # ------------------------------------------------------------------ #
    def viterbi(self, obs, verbose=True, digits=4):
        """Run the Viterbi algorithm.

        Returns
        -------
        path : tuple of str
            The most likely state sequence.
        logprob : float
            ``log`` of the joint probability of that path and the observations.
        """
        idx = self._to_indices(obs)
        t_max, n_states = len(idx), len(self.states)
        if t_max == 0:
            raise ValueError("the observation sequence is empty")

        with np.errstate(divide="ignore", invalid="ignore"):
            log_start = np.log(self.start_p)
            log_trans = np.log(self.trans_p)
            log_emit = np.log(self.emit_p)

        delta = np.full((t_max, n_states), -np.inf)
        back = np.zeros((t_max, n_states), dtype=int)

        delta[0] = log_start + log_emit[:, idx[0]]
        for t in range(1, t_max):
            for j in range(n_states):
                candidates = delta[t - 1] + log_trans[:, j]
                back[t, j] = int(np.argmax(candidates))
                delta[t, j] = candidates[back[t, j]] + log_emit[j, idx[t]]

        last = int(np.argmax(delta[t_max - 1]))
        path = [last]
        for t in range(t_max - 1, 0, -1):
            path.append(int(back[t, path[-1]]))
        path.reverse()

        logprob = float(delta[t_max - 1, last])

        if verbose:
            print("Viterbi trellis, probabilities of the best partial paths")
            print("t   " + "".join(f"{s:>12}" for s in self.states))
            for t, row in enumerate(delta):
                cells = "".join(
                    f"{v:>12.{digits}g}" if np.isfinite(v) else f"{0:>12.{digits}g}"
                    for v in np.exp(row)
                )
                print(f"{t + 1:<4}" + cells)
            print("\nbest path : " + " -> ".join(self.states[i] for i in path))
            print(f"log P(O, Q | lambda) = {logprob:.6f}")

        return tuple(self.states[i] for i in path), logprob

    # ------------------------------------------------------------------ #
    # convenience
    # ------------------------------------------------------------------ #
    def loglik(self, obs):
        """Shorthand for ``forward(obs)[1]``."""
        return self.forward(obs)[1]

    def posterior(self, obs):
        """Smoothed state probabilities ``P(q_t | O)`` from forward-backward."""
        alpha, loglik, scale = self.forward(obs)
        idx = self._to_indices(obs)
        t_max, n_states = len(idx), len(self.states)

        beta = np.ones((t_max, n_states))
        for t in range(t_max - 2, -1, -1):
            beta[t] = (self.trans_p @ (self.emit_p[:, idx[t + 1]] * beta[t + 1]))
            beta[t] /= scale[t + 1]

        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        return gamma

    def decode(self, obs, digits=6):
        """Print smoothed marginals next to the Viterbi path."""
        gamma = self.posterior(obs)
        path, _ = self.viterbi(obs, verbose=False)
        print("t   " + "".join(f"{s:>12}" for s in self.states) + f"{'viterbi':>12}")
        for t, row in enumerate(gamma):
            cells = "".join(f"{v:>12.{digits}g}" for v in row)
            print(f"{t + 1:<4}" + cells + f"{path[t]:>12}")
        return gamma, path
