# Why SDA Can Report False Convergence When $D < d(x^*)$

This note explains the negative `final_gap` / false `converged=True` behavior seen in
the nonsmooth objective grid, especially for runs such as:

```text
objective_id=max_affine_4d
method=sda_simple_euclidean
D=0.5
restrict_to_fd=False
```

The short version: the primal-dual SDA gap used as the stopping certificate is valid
only for a prox ball $F_D$ that contains an optimizer. If $x^*$ is outside that ball,
the quantity computed by the algorithm is no longer a certificate for the original
problem. It may even become negative, so a stopping rule like
$\delta_k(D) / S_k \le \varepsilon$ can fire even when the objective is not close to
optimal.

## Paper Setup

Nesterov defines a prox function $d$ with prox-center $x_0$, and the restricted prox
set:

$$
F_D = \{x \in Q : d(x) \le D\}.
$$

For dual averaging, with weights $\lambda_i$, subgradients $g_i$, and accumulated
weight $S_k$, the paper defines the primal-dual gap:

$$
\delta_k(D)
= \max_{x \in F_D}
  \sum_{i=0}^{k} \lambda_i \langle g_i, x_i - x \rangle.
$$

The implementation uses the equivalent computable form:

$$
\delta_k(D)
=
\sum_{i=0}^{k} \lambda_i \langle g_i, x_i - x_0 \rangle
+ \xi_D(-s_{k+1}),
$$

where

$$
s_{k+1} = \sum_{i=0}^{k} \lambda_i g_i,
\qquad
\xi_D(s) = \max_{x \in F_D} \langle s, x - x_0 \rangle.
$$

In `src/pda/sda.py`, this is:

```text
gap_accum += lambda_k * <g_k, x_k - x0>
gap       = gap_accum + xi_D(-s)
```

The normalized gap is:

$$
\operatorname{normalized\_gap}_k = \frac{\delta_k(D)}{S_k}.
$$

For simple dual averaging, $S_k = k + 1$. For weighted dual averaging, $S_k$ is the
sum of the adaptive weights.

## Required Radius Condition

The paper explicitly fixes $D$ so that:

$$
D \ge d(x^*).
$$

Equivalently:

$$
x^* \in F_D.
$$

This condition is not cosmetic. It is what makes $\delta_k(D)$ a valid certificate for
the original optimization problem.

When $x^* \in F_D$, the subgradient inequality gives:

$$
f(x_i) - f(x^*) \le \langle g_i, x_i - x^* \rangle.
$$

Since $x^*$ is one of the feasible choices in the max defining $\delta_k(D)$,

$$
\sum_i \lambda_i \langle g_i, x_i - x^* \rangle
\le
\max_{x \in F_D}
\sum_i \lambda_i \langle g_i, x_i - x \rangle
=
\delta_k(D).
$$

Combining this with convexity of $f$ gives the certificate:

$$
f(\hat{x}_k) - f(x^*) \le \frac{\delta_k(D)}{S_k}.
$$

This is the mathematical reason SDA can stop when:

$$
\frac{\delta_k(D)}{S_k} \le \varepsilon.
$$

## What Breaks When $D < d(x^*)$

If $D < d(x^*)$, then $x^* \notin F_D$. The proof above breaks at the key step:
we are no longer allowed to plug $x^*$ into the max over $F_D$.

The computed gap then certifies, at best, behavior relative to the restricted problem:

$$
\min_{x \in F_D} f(x).
$$

It does not certify the original unconstrained problem.

The paper also notes that for some $D$, $\delta_k(D)$ can be negative. The
nonnegativity guarantee requires $D \ge d(x^*)$. Once $x^*$ is outside $F_D$, a
negative computed gap is not paradoxical. It just means the quantity is not a valid
primal-dual certificate for the target problem.

Therefore this stopping check is unsafe when $D < d(x^*)$:

$$
\frac{\delta_k(D)}{S_k} \le \varepsilon,
$$

because a negative invalid certificate automatically satisfies the inequality.

## Concrete Example From This Repository

For the Euclidean prox used in `nonsmooth_objective_grid`:

$$
d(x) = \frac{1}{2}\lVert x - x_0 \rVert_2^2,
$$

and for registry objectives the prox-center is $x_0 = 0$.

For `max_affine_4d`, the known minimizer is:

$$
x^* =
\begin{bmatrix}
2.0 & -1.5 & 0.75 & 2.5
\end{bmatrix},
$$

with

$$
\lVert x^* \rVert_2 = 3.6142080737,
\qquad
d(x^*) = \frac{1}{2}\lVert x^* \rVert_2^2 = 6.53125.
$$

So with $D = 0.5$, the true minimizer is far outside the prox ball:

$$
d(x^*) = 6.53125 > 0.5 = D.
$$

The corresponding Euclidean radius is:

$$
\sqrt{2D} = 1,
$$

but

$$
\lVert x^* \rVert_2 \approx 3.61.
$$

The selected ball simply cannot contain the true optimizer.

This explains the observed run:

```text
converged = True
objective_gap ~= 1.626805
final_gap ~= -0.013874
```

Here `objective_gap` is the actual final objective gap at $\hat{x}$, so it correctly
says the run is not close to optimal.

But `final_gap` is the normalized SDA certificate:

$$
\operatorname{final\_gap}
=
\frac{\delta_k(D)}{S_k}.
$$

Because $D$ excludes $x^*$, this certificate is invalid for the original problem and
has gone negative. The code then marks the run converged because:

$$
-0.013874 \le \varepsilon.
$$

Mathematically, that is a false convergence signal.

## Restricted vs Unrestricted Iterates

There are two separate issues:

1. $D$ must be large enough: $D \ge d(x^*)$.
2. If the algorithm is interpreted as solving the restricted problem over $F_D$, the
   reported primal points should also be feasible for that restricted problem.

In the current experiment grid, `restrict_to_fd=False` means the primal iterates are
not projected back into $F_D$. The gap still uses the support function of $F_D$, but
the primal trajectory can move outside that set. This is especially confusing when
$D$ is too small: the gap is computed against a ball that does not contain the true
solution, while the displayed trajectory is not constrained to stay inside that ball.

When `restrict_to_fd=True`, the iterates are projected into $F_D$, but if
$D < d(x^*)$, the algorithm is solving or certifying only the restricted problem, not
the original unconstrained problem.

## Practical Rule

For registry objectives with known minimizer, choose:

$$
D \ge d(x^*)
$$

before trusting `converged`, `final_gap`, or `normalized_gap`.

For Euclidean prox centered at zero:

$$
D \ge \frac{1}{2}\lVert x^* \rVert_2^2.
$$

Examples:

```text
max_affine_4d:
  d(x*) = 6.53125
  D=0.5 and D=2 are invalid for the original problem.
  D=8 and D=32 contain x*.

ill_conditioned_max_affine_8d:
  d(x*) = 11.65625
  D=0.5, D=2, and D=8 are invalid for the original problem.
  D=32 contains x*.
```

## Implication For Plots And CSVs

Use these columns differently:

```text
objective_gap
objective_gap_x
objective_gap_x_hat
```

These compare against the known true objective optimum, so they reveal the false
convergence.

```text
final_gap
normalized_gap
converged
```

These are based on the SDA certificate. They are meaningful only when the prox ball
used in the certificate contains the true minimizer, and should be treated as invalid
when $D < d(x^*)$.

For plots, any run with $D < d(x^*)$ should either be filtered out when studying SDA
certificate convergence, or visually marked as having an invalid radius for the true
problem.

