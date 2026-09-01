# GridGuard — Ethics & Responsible Deployment

Electricity-theft detection is a **high-stakes, adversarial, human-impact**
domain. This project is a course/portfolio demonstration of production-grade
ML, and it ships with explicit constraints on how such a system may be used.

## 1. What this model can and cannot claim

- It estimates the probability that a consumption pattern resembles patterns
  of consumers **previously caught** stealing by a utility (SGCC). It does
  not "prove" theft.
- Labels come from inspections — the "honest" class contains undetected
  thieves, and flagged consumers may include closed accounts or clerical
  errors. Expect label noise; treat scores as evidence, never verdicts.
- No system, including this one, should be the **sole** basis for
  disconnecting a consumer's supply, fining them, or initiating prosecution.

## 2. Required human oversight

- Scores rank a **watchlist**; every field visit must be validated by a
  human inspector with physical evidence (meter sealing, load tests,
  account status).
- The web app displays, on every consumer detail: the tier, the plain-English
  explanation ("why flagged"), and honest-consumer bands for each feature —
  so a reviewer can audit the reason, not just the number.
- Recommended workflow: top-k% of scores → human desk review → field visit →
  evidence-based enforcement. The model's job ends at the watchlist.

## 3. Fairness considerations

- Consumption patterns correlate with income, household size, occupancy,
  season, and tariff class. A model like this can disproportionately flag
  low-consumption (poor) consumers if carelessly thresholded.
- Mitigations in this project: features are behavioural/shape-based (not
  raw magnitude alone); explanations show deviations vs **honest bands**
  so reviewers see when a "low usage" flag is really a poverty proxy;
  two thresholds let operators choose precision over recall when
  consequences are severe.
- The SGCC dataset carries **no demographic attributes** — a limitation:
  disparate impact could not be measured here. A production deployment in
  India (or elsewhere) must run bias audits on regional/tariff/phase slices
  before go-live, and periodically after.

## 4. Adversarial behaviour & safety

- Thieves adapt: partial-load theft (keeping apparent consumption plausible)
  is designed to evade exactly this class of detector. Expected counter:
  pair with physical checks (meter-vs-feeder energy balance, SEB comparisons).
- Publishing per-consumer scores publicly would help evasion; in deployment
  scores are internal, on a need-to-know basis.

## 5. Privacy

- The SGCC benchmark is fully anonymized (consumer numbers only, no PII,
  no addresses). The demo app scores **in-browser only** — the "analyze your
  own series" mode never uploads data anywhere (inference is local WASM).
- If deployed on real Indian DISCOM data: consent/notice obligations under
  applicable law (IT Act 2000 / DPDP Act 2023 for India), data minimization,
  and audit logs for every watchlist access are baseline requirements.

## 6. Non-goals

- Not a billing system; not a disconnect trigger; not surveillance of
  individuals beyond their own metered consumption.

## 7. Cited context for the problem

- Non-technical loss (NTL): theft via meter tampering, bypass, and billing
  fraud. Indian DISCOM AT&C (aggregate technical & commercial) losses are
  tracked by the Power Ministry's RDSS programme; aggregate losses of the
  order of ~20% historically cited for worst-hit utilities. This project
  demonstrates the *screening* layer of an NTL-reduction programme.
