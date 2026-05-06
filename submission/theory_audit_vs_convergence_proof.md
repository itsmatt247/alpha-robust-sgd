# Audit: `theory/convergence_proof.tex` vs codebase

This is reviewer-facing diligence: places where the written proof **already matches**, **almost matches**, or **does not yet** match implementation.

**Last updated 2026-05-05:** rigorous-pass fixes applied to address Tier-A audit findings (lagged-window contract, median sandwich, Theorem 1 statement, scope of Hill concentration). See "Rigorous-pass fixes" section at the bottom for the change log.

---

## Matches (good anchors)

1. **Fixed schedule role of \(\hat p\) in prose (NumPy-aligned).**  
   The draft explicitly notes that under **fixed**, \(\tau_t = C \hat\sigma_t\) depends on the **median scale**, and \(\hat p\) is partly **diagnostic** (`convergence_proof.md` Lemma 2 remark). That matches **NumPy** `AlphaRobustSGD`.

2. **Growing schedule uses \(\hat p\).**  
   Code and proof discussion agree \(\tau \propto t^{1/\hat p}\) modulates heaviness.

3. **Heavy-tail moment assumption (A2).** Bounded \(p\)-th moment noise is consistent with how Phase 1 **Pareto** noise is used experimentally.

4. **Burn-in with constant \(\tau_0\)** while Hill warms up—implemented in code.

---

## Divergences (must disclose or fix)

### ~~D1 — Torch `fixed` includes \((\hat p/2)\) factor~~ — **RESOLVED**

- **Proof** (`convergence_proof.tex`, §6 *Coupled Calibration*) now covers a general **Lipschitz tail-aware calibration** \(\tau_t = \varphi(\hat p_t)\,C\,\hat\sigma_t\):
  - **Theorem `thm:coupled`** — coupled adaptive fixed schedule, joint perturbation of \(\hat p_t\) and \(\hat\sigma_t\).
  - **Corollary `cor:half_index`** — specialization to the Torch implementation \(\varphi(q)=q/2\).
  - **Proposition `prop:joint_perturb`** — \(\tau_t/\tau_t^* \in [1\pm\mu_{\max}]\) with \(\mu_{\max}=\frac{L_\varphi}{\varphi_{\min}}\Delta+\epsilon+\frac{L_\varphi}{\varphi_{\min}}\Delta\,\epsilon\).
- **Rate** \(T^{-2(p-1)/(3p-2)}\) is **identical** to oracle fixed (Theorem `thm:fixed`) up to a vanishing \(O(\sqrt{\log T/W})\) overhead. Calibration map enters only through multiplicative constants (Remark `rem:phi_rate`).
- **Practical reading:** NumPy = \(\varphi\equiv 1\), Torch = \(\varphi(q)=q/2\) — both are valid algorithm choices under the same theorem.

### D2 — `deferred_hill` (Phase 3 PPO)

- Proof pipeline assumes \(\hat\sigma_t\) updated from a **streaming norm process** each step / standard filtration tricks.  
- PPO feeds **one median norm per optimizer epoch** into Hill (`flush_epoch_update`).

**Paper fix:** labeled **“engineering variant for correlated minibatches”**; convergence claim **excluding** defer unless you add a substantive dependent-data lemma.

### D3 — Lemma 1.1 / median concentration under heavy tails

Rolling **median** robustness is plausible as narrative, but the draft’s **Gaussian-style** \( \exp(-2W\delta^2/R^2) \)-type bound is **not** automatically valid for arbitrary heavy-tailed norm observations without extra structure. Reviewers may flag **soundness**.

**Mitigation paths:**

- Restrict lemma to bounded observations / truncation / noise domination regime used in Phase 1, **or**
- Replace with a known heavy-tail robust median bound with explicit assumptions **or**
- downgrade to **high-level remark** pending full proof.

### D4 — Lemma 1.2 sub-Gaussian tail claim for finite \(k\) Hill error

Classifier “\(P(|Hill-\alpha|>\varepsilon)\le 2\exp(-k\varepsilon^2/(2\alpha^2))\)” reads **strong**. EVT literature usually gives asymptotic normality regimes with **careful \(k,W\) interplay**. Needs **proper citation + conditions**, or softened.

### D5 — Lemma 3.3 conditioning & independence

Conditional on \(\hat\sigma_t\) measurable w.r.t. past norms is plausible, but **joint adaptation** across \(T\) requires **filtration bookkeeping** reviewers will check ( Nguyen-style care ).

---

## Minimum theorem package checklist (what “done” looks like)

- [ ] One **canonical algorithm box** labeled exactly which branch (NumPy vs Torch-fixed vs Torch-fixed-no-tail-factor).
- [ ] Statement: **nonconvex**: bound on \(\frac1T\sum \mathbb E[\|\nabla f\|^2]\) or equivalent—match cited baseline ( Zhang / Nguyen family ).
- [ ] Appendix: **complete** lemmas or clearly marked **conjectures** removed from main claims.
- [ ] Explicit **limitations** referencing D2–D4.

---

## Suggested wording for Related Work crossover

"Our primary guarantee targets the **median-scale clipped iterate** analyzed in clipped-SGD under heavy tails; learning \(\hat p\) affects **growth schedules** (\(\tau\propto t^{1/\hat p}\)) and a **coupled fixed-schedule calibration** \(\tau_t=\varphi(\hat p_t)\,C\,\hat\sigma_t\) for any bounded-Lipschitz \(\varphi\). The PyTorch implementation uses \(\varphi(q)=q/2\); the NumPy reference uses \(\varphi\equiv 1\); both are covered by Theorem (`thm:coupled`) at the same rate \(T^{-2(p-1)/(3p-2)}\). Oracle-free \(\hat p\) is novel relative to clipped analyses that assume known \(p\) for \(\tau_t\)."

---

## Rigorous-pass fixes (2026-05-05)

The following Tier-A audit findings were addressed in this revision. Each fix is paired with the touched code/theory locations.

### F1 — Lagged-window contract (was D2/A1)

**Problem.** Both the NumPy and Torch optimizers updated the rolling window (and Hill estimator) with the **current** step's gradient norm **before** computing \(\tau_t\). Under the proof's filtration \(\mathcal F_t = \sigma(x_1, \varepsilon_1, \ldots, \varepsilon_{t-1})\), this made \(\tau_t \notin \mathcal F_t\), invalidating the bias/variance bounds in Lemmas 2–3 (which assume \(\tau\) measurable when conditioning by \(\mathcal F_t\)).

**Fix.**
- **Code (NumPy `optimizers/alpha_robust_sgd.py:step`)**: window/Hill update moved to **after** \(\tau_t\) is used.
- **Code (Torch `optimizers/torch/alpha_robust_sgd.py:step`)**: same; deferred-Hill mode also lagged for consistency.
- **Theory (`convergence_proof.tex` §1)**: algorithm description now states the lagged-window contract explicitly; added `Remark rem:lag` explaining why measurability requires it.
- **Theory (Theorems 2/3/4)**: opening step of each adaptive proof now invokes `Remark rem:lag` to justify \(\tau_t \in \mathcal F_t\).

**Verification.** The four AlphaRobust unit tests (convergence, p-hat tracking, log lengths, no-NaN) pass after the change.

### F2 — Theorem 1 \(\tau^*(T)\) display vs. derivation (was A2)

**Problem.** The theorem statement displayed \(\tau^*\) without \(L\) or \(\Delta_f\) dependence, but the proof's eq. \((10)\) `eq:tau_opt` shows the optimum *does* depend on \(L\Delta_f\). This was an internal inconsistency.

**Fix.** Theorem 1's display in `eq:optimal_rate` now matches `eq:tau_opt` exactly, with the \(A_B / A_V\) abbreviation defined.

### F3 — Lemma 5 "CDF horizontal shift" (was A3)

**Problem.** The original lemma claimed the CDF of \(\|g_s\|\) is a horizontal shift of the CDF of \(\|\varepsilon_s\|\) by at most \(M\). This is **false in \(d > 1\)**: adding a deterministic vector \(b\) to \(\varepsilon_s\) changes the law of \(\|\cdot\|\) in a direction-dependent way, not as a location family.

**Fix.**
- **Replaced** the false shift claim with the (correct) **deterministic norm sandwich** \(\|\varepsilon_s\| - M \leq \|g_s\| \leq \|\varepsilon_s\| + M\).
- **Added** new Lemma `lem:median_sandwich` (Hoeffding for non-i.d. medians with approximate-median band of width \(M_0\)).
- **Added** `Remark rem:no_shift` explaining why the previous argument was wrong.
- **Added** explicit **slab-density hypothesis** to Lemma `lem:conditioning(c)` with sufficient conditions in `Remark rem:slab_alternatives` (Pareto-spheroidal noise, etc.).

### F4 — Lemma 4 [sic] constant (was A4)

**Problem.** Inline annotation `[sic]` flagged a factor-of-4 mismatch between the displayed exponent \(2\exp(-W\delta^2/(2R^2))\) and Hoeffding's \(2\exp(-2W(c\delta)^2)\) with \(c=1/R\).

**Fix.** Lemma `lem:median` now correctly displays \(2\exp(-2W\delta^2/R^2)\); the `[sic]` annotation is removed. Proposition `prop:scale` updated to use \(2\exp(-2W c_{\mathrm{cdf}}^2 \delta^2)\) with the explicit slab-density constant.

### F5 — Proposition 1 non-common-median bridge (was B1)

**Problem.** The original proof was 3 lines: "by Lemma 5, apply Lemma 4." But Lemma 4 requires a **single common** \(m\) across the \(W\) variables, while Lemma 5 only gives each marginal median in a **band** of width \(M\) around \(\sigma_{\mathrm{med}}\).

**Fix.** New Lemma `lem:median_sandwich` handles the non-i.d. median case directly: condition on \(m_i \in [m_0 - M_0, m_0 + M_0]\) (band) plus uniform local-slope condition on a slab of width \(M_0 + R\) around \(m_0\). Proposition 1's proof now invokes this lemma cleanly.

### F6 — Unbiased noise in Assumption (was A5)

**Fix.** Assumption 2 (heavy-tailed noise) now explicitly includes \(\mathbb E[\varepsilon_t \mid \mathcal F_t] = 0\) as part (a), with conditional independence as part (c). Earlier draft only stated bounded \(p\)-th moment.

### F7 — Hill concentration scope (was B2)

**Fix.** `Remark rem:beyond_pareto` now states explicitly:
- Lemma 7 applies under **exact i.i.d. Pareto** (Phase 1).
- For non-Pareto regularly-varying tails (Phase 2/3), the constant rate is replaced by an asymptotic \(A(n/k)\) bias and the deterministic clamping bound is the operative finite-sample guarantee.
- Proposition 3's union bound now offers two scopes: an exponential bound under exact Pareto, or the clamping fallback for general tails (in which case Theorem 3 converges to a neighborhood, not the rate).

### F8 — Growing-schedule honest known-\(T\) caveat (was B4)

**Fix.** Theorem 4 now:
- **Explicitly states** \(\eta_0^* \propto T^{(p-2)/(2p)}\) depends on \(T\),
- **Explicitly clarifies** that the oracle threshold uses \(\sigma_{\mathrm{med}}\) (not the moment scale \(\sigma\)), absorbing the constant \(\sigma_{\mathrm{med}}/\sigma\) into \(C'\),
- **Lists two unknown-\(T\) escape hatches**: doubling trick and an alternative LR schedule \(\eta_t = \eta_0 / t^{p/(2p-2)}\).

---

## Remaining open items (not Tier-A; flagged for future work)

- **Median sandwich constant.** The slab-density hypothesis on \(\|g_s\|\) (Lemma `lem:conditioning(c)`) is verified for Pareto-spheroidal noise but does not follow from Assumption `ass:tail` alone. For settings outside Phase 1, this is an additional standing assumption.
- **Hill non-i.i.d. concentration.** A finite-sample concentration bound on the Hill estimator under non-Pareto regularly-varying inputs is an open problem in the EVT literature; we use the deterministic clamping fallback for Phase 2/3.
- **Deferred-Hill (PPO) mode.** The proof pipeline assumes streaming Hill updates per step; the `flush_epoch_update` deferred mode in PPO is an engineering variant that we do not formally cover.
