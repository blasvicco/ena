# Editorial Peer Review — Evolutionary Neuroplastic Agent

Independent assessment | Q1/Q2 journal standards | Manuscript: Mitigating Catastrophic Forgetting with Physics-Indexed Specialist Memory: the Evolutionary Neuroplastic Agent | 23 pages

## 1. Executive editorial verdict

Major Revision

Current readiness: Borderline Q2 · Not Q1-ready

The manuscript presents a credible experimental contribution, but its central causal claim is not yet sufficiently isolated. The revised framing is substantially more defensible than claiming superiority of neuroevolution: the paper asks whether failure-driven search, specialist preservation, and environment detection jointly mitigate catastrophic forgetting. Its CartPole ablations provide meaningful evidence, supported by multiple replicates, effect sizes, confidence intervals, and multiplicity correction.

However, three weaknesses remain serious: the archive ablation changes both memory preservation and test-time policy selection; the Plasticity comparisons use substantially different interaction budgets; and the strongest continual-learning alternatives are discussed but not evaluated. These issues do not invalidate the observed results, but they limit what can be concluded from them.

Editorial decision: Major Revision for a suitable Q2 journal, potentially publishable after stronger controlled experiments. For a selective Q1 journal, I would expect a high risk of rejection in the present form, primarily because of incomplete causal validation and limited evidence of generality, rather than poor writing or an obviously unsound concept.

## 2. High-level editorial assessment

| Criterion               | Assessment                                                                                           |
| ----------------------- | ---------------------------------------------------------------------------------------------------- |
| Originality             | Moderate — the integration is distinctive, but most individual mechanisms have precedents            |
| Scientific significance | Promising, particularly the interaction between detection and archived policy reuse                  |
| Technical soundness     | Generally credible, with important experimental confounds                                            |
| Experimental rigor      | Above average in statistical reporting; incomplete in causal controls                                |
| Reproducibility         | Good methodological detail; executable reproducibility claims still require independent verification |
| Generalizability        | Limited to two benchmark families and fixed training sequences                                       |
| Contribution level      | Moderate, primarily architectural and empirical                                                      |
| Q1/Q2 interest          | Relevant to continual RL, neuroevolution, and adaptive autonomous systems                            |

The strongest aspect of the paper is its attempt to answer specific research questions rather than simply introduce another agent architecture. The distinction between learning an environment and retaining competence after leaving it is scientifically important.

The novelty must nevertheless be positioned carefully. Physics-indexed policy archives overlap conceptually with context-based policy preservation, while failure-responsive evolutionary search and Bayesian policy reuse are established ideas. The contribution is therefore the particular integration and its experimental characterization, not the invention of a fundamentally new continual-learning mechanism.

## 3. Technical peer review

### 3.1 Problem formulation and research questions

The formulation as a sequence of MDPs sharing observation space, action space, and reward function but differing in transition dynamics is coherent. RQ1 and its subquestions are explicitly mapped to experiments, which strengthens the paper.

The main conceptual issue is that preserving a frozen specialist and successfully retrieving that specialist are different capabilities. The paper acknowledges this distinction, but its principal ablation does not fully separate them.

There is also an important distinction between catastrophic forgetting at the policy level and degradation of the agent's overall performance. ENA can retain old policies without maintaining the performance of the currently evolving population. This is a valid architectural approach, but the paper should consistently describe the outcome as system-level retention through specialist preservation and retrieval.

### 3.2 Experimental design and controls

Concern A — Archive and selection are confounded

The no-archive variants deploy a random member of the final population at test, whereas archive-equipped variants use Bayesian policy reuse. Consequently, the experiment changes both what is retained and how behavior is selected.

This is the strongest causal-validity concern in the manuscript.

The separate selection ablation helps: Bayesian policy reuse achieves 83.2% reliability, versus 59.2% for niche matching and 53.8% for highest Trust. But that experiment isolates the choice among archived policies, not the independent contribution of archiving.

Required correction: Add a control that selects the best available specialist from the final population using a comparable selection protocol, without persistent archival protection. Ideally, also compare an archive with and without niche indexing under a common retrieval mechanism.

Concern B — Plasticity is not evaluated at equal interaction

The CartPole skill ablation fixes 500 episodes per environment. The Plasticity variants consume approximately 4.5–4.9 million environment steps, compared with roughly 0.27–0.34 million for the low-search variants.

Therefore, improved learning cannot be attributed exclusively to failure-responsive allocation. Much of the advantage may result from spending more interactions.

The single-replicate full-search experiment is informative but insufficient to establish efficiency.

Required correction: Compare failure-driven Plasticity against fixed-effort and alternative effort-allocation strategies under the same total interaction budget. Report both retention and achieved performance.

Concern C — Limited validation of the complete system

The complete skill ablation is performed on CartPole, while HalfCheetah primarily demonstrates retention against conventional single-policy learners and evaluates the detector separately.

The HalfCheetah results therefore support the complete architecture's behavior, but not the necessity or interaction of all its components in that domain.

A reduced HalfCheetah ablation would materially strengthen the manuscript.

### 3.3 Statistical rigor

This is one of the stronger aspects of the work. The paper uses 30 replicates for the principal skill ablation, 50 for the CartPole learner comparison, and 20 for HalfCheetah. It reports effect sizes, confidence intervals, Welch tests, Holm correction for planned comparisons, and power calculations.

Nevertheless, several issues remain:

- The same random seeds are used across variants, but Welch's unpaired test does not exploit this pairing. A paired analysis, where the experimental units are meaningfully matched, could improve precision.
- The controller comparison is underpowered for moderate effects. The quadratic controller achieves 89.3% reliability against the ramp's 83.2%, but the manuscript appropriately avoids claiming superiority.
- Retention ratios can be misleading when the training-phase denominator is low or includes exploration. Absolute forgetting helps, but does not fully resolve the issue.
- Multiple exploratory and per-environment comparisons are reported. Their inferential status must remain explicit throughout the discussion.

I would not reject the manuscript for its statistical methodology alone. Its principal weakness is experimental identification, not a general absence of statistical discipline.

### 3.4 Reproducibility and transparency

The manuscript provides considerable algorithmic detail: network architectures, evolutionary operators, detector equations, policy-selection rules, budgets, seeds, and implementation settings.

This is encouraging. However, the claim that all results are generated from released code and experiment outputs must be independently verifiable. I have not executed the repository or reproduced the reported results.

A particularly important reproducibility issue is the use of resettable copies of the current environment for candidate evaluation. The paper states that the agent is not informed of the physics, but its evaluation mechanism requires access to a simulator configured with the current physics.

This distinction should be made explicit in the problem assumptions:

The policy and detector are not given physics labels, but the training infrastructure has access to a resettable simulator of the current dynamics.

That does not invalidate the experiment. It restricts the deployment setting and changes how readers should interpret claims about autonomous adaptation.

### 3.5 Threats to validity

| Threat               | Severity      | Assessment                                                                                          |
| -------------------- | ------------- | --------------------------------------------------------------------------------------------------- |
| Internal validity    | High          | Archive/retrieval confound; unequal ablation budgets                                                |
| Construct validity   | Moderate–High | Retention measures system-level preservation, not necessarily policy-level resistance to forgetting |
| External validity    | High          | Two benchmark families, fixed task order, no training-time recurrence                               |
| Statistical validity | Moderate      | Strong reporting, but some underpowered comparisons                                                 |
| Reproducibility      | Moderate      | Detailed specification, but execution not independently verified                                    |

The most consequential missing experiment is recurrence during training. Earlier environments return during frozen evaluation, but the system is not tested under repeated environmental switches while learning continues.

That is particularly important for a continual-learning architecture designed to recall specialists and resume adaptation.

## 4. Section-by-section review

| Section      | Assessment and required revision                                                                                                                                                                                                                                |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Title        | Accurate and appropriately focused. However, “mitigating catastrophic forgetting” should be understood as system-level retention through policy retrieval. No essential change required.                                                                        |
| Abstract     | Clear, balanced, and unusually transparent about limitations. Keep the explicit statement that ENA does not claim superiority over continual-learning methods.                                                                                                  |
| Introduction | Strong research-question structure. The novelty argument is credible but needs a sharper explanation of what the integration enables beyond established context-based archives.                                                                                 |
| Related Work | Good coverage of EWC, CLEAR, OWL, RL-CD, MBCD, SWOKS, ITE, TeLAPA, and continual neuroevolution. The closest methods are correctly acknowledged. A stronger mechanism-level comparison would help establish the precise contribution.                           |
| Methodology  | Technically detailed and largely reproducible. The separation between archive admission, context identification, and retrieval should be made more explicit. The simulator-access assumption deserves greater prominence.                                       |
| Results      | Strong quantitative reporting. However, the principal ablation does not isolate all causal mechanisms. The HalfCheetah results should not be interpreted as a complete validation of the skills.                                                                |
| Discussion   | One of the manuscript's strongest sections. It explicitly acknowledges major experimental confounds. The distinction between demonstrated findings and possible explanations is generally well maintained.                                                      |
| Limitations  | Substantive rather than cosmetic. It identifies fixed task order, lack of training recurrence, simulator resets, and test-time selection confounds. These limitations are sufficiently important that some should become experiments before journal submission. |
| Conclusion   | Appropriately cautious. Avoid strengthening the causal language beyond what the ablation establishes.                                                                                                                                                           |
| References   | Relevant and technically appropriate. Verify all recent references and bibliographic metadata before submission. The citation to Nisioti et al. (2026) is particularly relevant to the novelty positioning.                                                     |

## 5. Major concerns

M1

The archive contribution is not causally isolated.

The no-archive condition changes the retrieval mechanism. A reviewer cannot determine how much improvement comes from retaining specialists versus selecting them more effectively.

Disposition: Remediable, but potentially rejection-level without additional controls.

M2

The efficiency claim for failure-driven search remains unproven.

Plasticity increases interaction substantially, and the fixed-full-search comparison has only one replicate. The current evidence establishes the benefit of greater search effort, not the optimality or efficiency of its allocation.

Disposition: Remediable through equal-budget experiments.

M3

No direct comparison against context-aware continual-learning systems.

The manuscript's explanation for excluding them is reasonable as a statement about experimental confounding. However, this does not eliminate the need for competitive context.

A Q1 reviewer may argue that the paper demonstrates a known benefit of keeping separate policies for separate contexts.

Disposition: Remediable; particularly important for Q1.

M4

The continual-learning setting is too restricted.

There is no recurrence during training, no randomized task ordering, and no experiment with repeated short-duration regimes. These are precisely the situations in which context recognition and policy recall should provide value.

Disposition: Remediable and a high-priority extension.

M5

HalfCheetah demonstrates retention of relatively weak policies.

ENA achieves 115.6% retention, compared with 39.2% for PPO and 34.4% for SAC. However, SAC achieves much higher absolute returns on most environments, including 10,146 versus ENA's 1,773 on Mars-Slippery.

The retention result is valid, but preservation of low-performing policies is not sufficient evidence of a competitive continual-learning system.

Disposition: Remediable through stronger within-architecture controls and explicit competence-aware evaluation.

## 6. Minor concerns

- Use Plasticity and Trust consistently as defined algorithmic quantities, rather than implying biological equivalence.
- Clarify the difference between policy weights being frozen and test-time beliefs and performance estimates continuing to adapt.
- Report detector false positives, false negatives, detection delays, and niche fragmentation prominently rather than relying on appendix tables.
- Separate environment-step efficiency from wall-clock efficiency throughout.
- Improve figure readability, especially where long variant names make horizontal labels difficult to interpret.
- Explain how the archive capacity and replacement threshold affect long-term scalability.
- Clarify how the reference return used for retention differs from the evaluation of the best available specialist.

These are secondary issues and should not independently prevent publication.

## 7. Hidden reviewer objections

The following are objections I would expect from demanding reviewers, even though the manuscript anticipates some of them.

Objection 1: “Is this catastrophic-forgetting mitigation or simply a collection of frozen policies?”

The system preserves earlier behavior by storing policies that are not modified. This is a legitimate solution, but a reviewer may consider the result unsurprising unless the paper demonstrates that online identification and retrieval are substantially better than simpler memory architectures.

Objection 2: “Does the environment detector really operate without privileged information?”

The detector uses observations, but the search mechanism relies on a simulator with the current physics. The distinction between what the agent observes and what the experimental infrastructure provides must be explicit.

Objection 3: “Is the HalfCheetah retention advantage partly an evaluation artifact?”

The end-of-phase return includes exploration, while frozen testing selects confirmed specialists. The two quantities may represent different policy-selection distributions. This could explain retention above 100% without requiring any improvement in underlying competence.

A matched-policy evaluation at phase end and final test would address this.

Objection 4: “Were detector configurations selected using the same environments on which their effectiveness is demonstrated?”

The HalfCheetah chirp design was selected using the three training environments, with a separate confirmation evaluation. This is not automatically leakage, but it does not demonstrate generalization to new families of dynamics. Testing detector configurations on unseen physics would strengthen the result.

Objection 5: “Why should the reader prefer ENA over a simpler context-indexed policy bank?”

This is potentially the most damaging novelty objection. A lightweight control combining a dynamics detector, a bank of independently trained policies, and a common selection mechanism would provide a direct reference for the value of ENA's evolutionary population and archive-management design.

## 8. Journal readiness assessment

CURRENT ASSESSMENT

# Borderline Q2

A plausible journal contribution, but not yet supported by sufficiently controlled evidence for a strong Q1 submission.

The top three reasons the paper would struggle are the archive/retrieval confound, the absence of an equal-budget test of failure-driven search, and the lack of demonstrated recurrence during training.

The top three changes most likely to improve acceptance are a controlled archive experiment, an equal-interaction Plasticity experiment, and a repeated-switch evaluation with randomized regime order.

## 9. Concrete revision roadmap

Critical revisions

0/3 completed

Isolate memory from retrieval

Compare archived and non-archived specialists under equivalent test-time selection rules.

Control interaction budgets

Run fixed-effort and adaptive-effort search under matched environment-step budgets across multiple seeds.

Evaluate training-time recurrence

Introduce sequences such as A–B–C–A–B–C and randomized orders, measuring recovery and retention.

Important revisions

0/4 completed

Extend the skill ablation

Repeat the most informative archive and detection controls on HalfCheetah.

Introduce a simple policy-bank comparator

Use an online context-indexed memory with a common detector and retrieval rule.

Audit retention metrics

Evaluate the same archived or selected policy at phase end and final test, alongside the current agent-level metrics.

Test detector generalization

Evaluate chirp identification on physics configurations not used in selecting its excitation and features.

Optional revisions

0/3 completed

Refine statistical analysis

Add paired estimates and uncertainty intervals where seed matching supports them.

Study sensitivity

Vary archive capacity, detection thresholds, and specialist population size.

Improve presentation

Consolidate figures, clarify terminology, and move key detector diagnostics into the main results.

## 10. Final recommendation to the authors

The manuscript is technically serious, substantially more disciplined than many architecture-driven reinforcement-learning papers, and correctly limits its central claim to the contribution of the combined skills rather than asserting that neuroevolution is inherently superior.

Its most convincing result is the CartPole evidence that failure-driven search enables acquisition of difficult environments, while physics-indexed specialist retrieval helps preserve previously acquired competence. The reported differences are large enough to merit scientific attention.

The remaining problem is not primarily writing quality. It is whether the experiments establish the causal contribution of the proposed mechanisms strongly enough to distinguish ENA from simpler context-indexed policy-memory systems.

I would not recommend rewriting the manuscript or fundamentally redesigning ENA. I would recommend strengthening three experiments before submission: isolate the archive effect, control search budgets, and demonstrate recurrence during training. Those additions could move this from a borderline Q2 submission to a credible Q2 candidate and improve its prospects at more selective journals.

The work has publishable potential. Its current evidence is promising, but not yet conclusive enough for a high-impact Q1 journal.
