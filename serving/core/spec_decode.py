"""Simulator-side model of speculative decoding acceptance.

Speculative decoding proposes ``num_speculative_tokens`` (N) draft tokens per
step and verifies them in one forward pass. The target model accepts a prefix
of them and always emits one bonus token, so a request advances by
``1 + accepted`` tokens per step instead of 1.

**Which tokens get accepted is the one thing a simulator cannot compute** -- it
needs the draft's and the target's actual distributions over the real
vocabulary, i.e. tokens and weights. So acceptance is a *policy*, chosen the
way MoE expert routing is (see ``gate_function.GateRouter``), with the default
taken from what the model's own authors published.

Acceptance rate semantics
-------------------------
The rate here is **marginal**: the fraction of draft steps in which a given
position was accepted, already including the probability of reaching it. So

    mean_accept_length = 1 + sum(per-position marginals)
                       = 1 + rate * N                      (constant marginals)

This is what the published sources report, and the identity reproduces all
nine (rate, length) pairs in ``configs/spec_decode.json`` to within 0.01
tokens.

It is deliberately **not** Leviathan's alpha (ICML 2023, arXiv:2211.17192),
which is *conditional* -- position i is reached only if 1..i-1 were accepted --
and gives the capped geometric ``(1 - a^(N+1)) / (1 - a)``. Passing a published
rate to that formula under-predicts the published accept length by 25-30%
(MiniMax-M3 2.42 against 3.0; Qwen 3.51 against 4.89), because real acceptance
is front-loaded rather than i.i.d. The check that settles it: Qwen's published
per-position decline of 95% at p1 to 60% at p5 averages 0.775 read as
marginals against a published 0.779, and 0.621 read as conditionals.

Per-position curves
-------------------
vLLM accepts a **prefix** of the draft: a step that accepts k tokens accepted
positions 1..k, and its counters record it that way
(``vllm/v1/spec_decode/metrics.py``, ``SpecDecodingStats.observe_draft``). So
the per-position marginals m_i fix the whole distribution of the accepted
count, ``P(accepted >= i) = m_i``, and a valid curve is nonincreasing. That is
the rule vLLM applies to its own ``synthetic_acceptance_rates``
(``vllm/config/speculative.py``): N entries, each in [0, 1], never rising.
``DECAY`` samples that distribution with one uniform draw per step.
"""

import json
import os
import random

from .logger import get_logger

_TABLE_CACHE = {}


def _table_path():
    base = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(base))
    return os.path.join(repo_root, "configs", "spec_decode.json")


def published_defaults(model_name):
    """The published entry for ``model_name``, or None.

    Absent means no figure could be found for that model, not that speculative
    decoding is unavailable -- the caller must then be given an explicit rate
    rather than a guessed one.
    """
    path = _table_path()
    if path not in _TABLE_CACHE:
        try:
            with open(path, "r", encoding="utf-8") as f:
                _TABLE_CACHE[path] = json.load(f).get("models") or {}
        except FileNotFoundError:
            _TABLE_CACHE[path] = {}
    return _TABLE_CACHE[path].get(model_name)


def check_position_acceptance(curve, num_speculative_tokens=None,
                              source="per-position acceptance"):
    """``curve`` as a list of floats, or ValueError.

    A per-position curve holds the marginal P(accepted >= i) for i = 1..N, so
    it must be nonincreasing with every entry in [0, 1], and hold N entries
    when N is known. The same rule vLLM applies to
    ``synthetic_acceptance_rates``. Nothing is clipped: a curve that breaks it
    describes no prefix acceptance, and silently repairing it would simulate a
    different one.
    """
    if isinstance(curve, (str, bytes)) or not hasattr(curve, "__iter__"):
        raise ValueError(f"{source} must be a list of rates; got {curve!r}")
    values = []
    for p in curve:
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            raise ValueError(f"{source} entries must be numbers; got {curve!r}")
        values.append(float(p))
    if not values:
        raise ValueError(f"{source} must hold at least one rate")
    if num_speculative_tokens is not None and len(values) != num_speculative_tokens:
        raise ValueError(
            f"{source} has {len(values)} entries but num_speculative_tokens is "
            f"{num_speculative_tokens}; it needs one rate per draft position")
    if any(not 0.0 <= p <= 1.0 for p in values):
        raise ValueError(f"{source} entries must be in [0, 1]; got {values}")
    if any(later > earlier for earlier, later in zip(values, values[1:])):
        raise ValueError(
            f"{source} must be nonincreasing, since position i is accepted only "
            f"when every earlier position was; got {values}")
    return values



class AcceptanceModel:
    """How many of a step's N draft tokens the target model accepts.

    Policies:
        FIXED   (default) -- every position accepted with the same marginal
                             probability, so E[accepted] = rate * N and the
                             mean accept length is exactly the published one.
        DECAY             -- the accepted count is a prefix of the draft, as in
                             vLLM: one uniform draw u per step, and position i
                             is accepted while u < m_i, so P(accepted >= i) =
                             m_i. The curve m is the model's published
                             ``position_acceptance``, else a linear ramp with
                             the same mean as the aggregate rate. Acceptance
                             really does fall with draft position
                             (Qwen3.5-27B: 95% at p1 to 60% at p5). On that
                             published curve the prefix draw has variance 2.61
                             against FIXED's 0.86 at nearly the same mean, so
                             the spread differs, not only the shape.
        CUSTOM            -- user-supplied ``_custom_acceptance_function``.

    Acceptance is drawn, not averaged, because a scheduler is not a mean: two
    requests accepting 0 and 4 tokens occupy different numbers of slots and
    finish at different times from two accepting 2 each.
    """

    _SUPPORTED_POLICIES = ("FIXED", "DECAY", "CUSTOM")

    def __init__(
        self,
        num_speculative_tokens,
        acceptance_rate,
        position_acceptance=None,
        policy="FIXED",
        seed=42,
        node_id=0,
        instance_id=0,
    ):
        self.N = max(int(num_speculative_tokens), 0)
        self.rate = min(max(float(acceptance_rate), 0.0), 1.0)
        self.policy = str(policy).upper()
        self.rnd = random.Random(seed) if seed is not None else random
        self.logger = get_logger(
            self.__class__, node_id=node_id, instance_id=instance_id
        )

        if self.policy not in self._SUPPORTED_POLICIES:
            raise ValueError(
                f"[AcceptanceModel] unsupported policy {policy!r}; "
                f"expected one of {self._SUPPORTED_POLICIES}"
            )

        self.position_acceptance = self._resolve_positions(position_acceptance)

        if self.policy == "DECAY":
            self.accept_fn = self._decay
        elif self.policy == "CUSTOM":
            self.accept_fn = self._custom_acceptance_function
        else:
            self.accept_fn = self._fixed

    def _resolve_positions(self, published):
        """Per-position marginals of length N, summing to ``rate * N``.

        A supplied curve is used as given when it is the right length, and
        refused when it is not a valid prefix curve (see
        ``check_position_acceptance``). With only an aggregate, the curve is a
        linear ramp whose mean is that aggregate -- the shape the one source
        that publishes both reports, rather than a flat line, since a flat line
        is what FIXED already is.
        """
        if self.N == 0:
            return []
        if published and len(published) == self.N:
            return check_position_acceptance(published, self.N)
        if self.N == 1:
            return [self.rate]
        # Ramp from rate + spread to rate - spread, clipped to [0, 1]. The
        # spread is the published Qwen decline (0.95 -> 0.60) scaled to this
        # rate, and shrinks near the ends so the mean survives clipping.
        spread = min(0.175, self.rate, 1.0 - self.rate)
        step = 2.0 * spread / (self.N - 1)
        return [self.rate + spread - step * i for i in range(self.N)]

    # -------------------- policies --------------------

    def _fixed(self, n_draft):
        """Each position accepted independently at the pooled marginal rate."""
        accepted = 0
        for _ in range(n_draft):
            if self.rnd.random() < self.rate:
                accepted += 1
        return accepted

    def _decay(self, n_draft):
        """A prefix of the draft: one uniform draw, P(accepted >= i) = m_i.

        The curve is nonincreasing, so the positions with u < m_i are exactly
        the first k of them. A step that drafts fewer than N tokens (its draft
        clamped to the step's token budget) uses the first ``n_draft`` rates.
        """
        u = self.rnd.random()
        accepted = 0
        for p in self.position_acceptance[:n_draft]:
            if u >= p:
                break
            accepted += 1
        return accepted

    def _custom_acceptance_function(self, n_draft):
        """Override point. Return how many of ``n_draft`` are accepted.

        Left as the pooled rate so an unedited CUSTOM run behaves like FIXED
        rather than silently accepting everything.
        """
        return self._fixed(n_draft)

    # -------------------- API --------------------

    def draw(self, n_draft):
        """Accepted count for one verification step, in ``[0, n_draft]``."""
        n_draft = max(int(n_draft), 0)
        if n_draft == 0:
            return 0
        return max(0, min(int(self.accept_fn(n_draft)), n_draft))

    def mean_accept_length(self):
        """Expected tokens committed per step, including the bonus token."""
        if self.N == 0:
            return 1.0
        if self.policy == "DECAY":
            return 1.0 + sum(self.position_acceptance)
        return 1.0 + self.rate * self.N


def _selftest():
    """Reproduce every published (rate, length) pair from configs/."""
    import statistics

    table = json.load(open(_table_path()))["models"]
    print("model                            N   rate   published  FIXED   DECAY")
    for name, e in table.items():
        N, rate, pub = (e["num_speculative_tokens"], e["acceptance_rate"],
                        e["mean_accept_length"])
        pos = e.get("position_acceptance")
        fixed = AcceptanceModel(N, rate, pos, policy="FIXED")
        decay = AcceptanceModel(N, rate, pos, policy="DECAY")
        assert abs(fixed.mean_accept_length() - pub) < 0.02, name
        assert abs(decay.mean_accept_length() - pub) < 0.05, name
        # drawn mean must match the closed form
        drawn = statistics.mean(1 + fixed.draw(N) for _ in range(20000))
        assert abs(drawn - fixed.mean_accept_length()) < 0.05, (name, drawn)
        print(f"{name:<32}{N:>2}{rate:>7.3f}{pub:>11.2f}"
              f"{fixed.mean_accept_length():>8.2f}{decay.mean_accept_length():>8.2f}")

    # bounds
    m = AcceptanceModel(4, 0.5)
    assert all(0 <= m.draw(4) <= 4 for _ in range(1000))
    assert AcceptanceModel(0, 0.9).draw(0) == 0
    assert AcceptanceModel(0, 0.9).mean_accept_length() == 1.0
    # a published curve is used verbatim
    q = table["Qwen/Qwen3.8-27B"]
    assert AcceptanceModel(5, q["acceptance_rate"],
                           q["position_acceptance"]).position_acceptance \
        == q["position_acceptance"]

    # DECAY draws a prefix: P(accepted >= i) is the curve, not a product
    curve = q["position_acceptance"]
    decay = AcceptanceModel(5, q["acceptance_rate"], curve, policy="DECAY")
    draws = [decay.draw(5) for _ in range(200000)]
    for i, m in enumerate(curve, start=1):
        share = sum(d >= i for d in draws) / len(draws)
        assert abs(share - m) < 0.005, (i, share, m)
    assert abs(statistics.mean(draws) - sum(curve)) < 0.02
    # a clamped step uses the first n_draft rates
    assert all(0 <= decay.draw(2) <= 2 for _ in range(1000))
    short = [decay.draw(2) for _ in range(100000)]
    assert abs(sum(d >= 2 for d in short) / len(short) - curve[1]) < 0.01

    # curve validation, vLLM's rule for synthetic_acceptance_rates
    for bad, n in (([0.5, 0.6], 2), ([1.2, 0.5], 2), ([-0.1], 1),
                   ([0.9, 0.8], 3), ([], None), ("0.9", None), ([True], 1)):
        try:
            check_position_acceptance(bad, n)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid curve {bad!r} for N={n}")
    assert check_position_acceptance([0.9, 0.9, 0.5], 3) == [0.9, 0.9, 0.5]
    try:
        AcceptanceModel(2, 0.5, [0.4, 0.6], policy="DECAY")
    except ValueError:
        pass
    else:
        raise AssertionError("a rising published curve was accepted")

    print("spec_decode self-test: all checks passed")


if __name__ == "__main__":
    _selftest()
