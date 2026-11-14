### 1. Executive editorial verdict

* **Preliminary Decision:** **Major Revision** (Borderline Reject)
* **Rationale:** The manuscript presents a conceptually intriguing and thorough investigation into gradient-free continual reinforcement learning via the Evolutionary Neuroplastic Agent (ENA), combining failure-driven search, physics-indexed memory (Hall of Fame), and Bayesian policy reuse. The empirical evaluations—spanning extensive skill ablations, controller variations, selection strategies, and a dedicated detection benchmark—are commendable for their transparency and rigor. However, the manuscript suffers from a critical methodological limitation: it dismisses direct comparison against state-of-the-art continual reinforcement learning baselines (such as MBCD, SWOKS, or TeLAPA) under the premise that they use gradient training or require task boundaries. In a premier Q1 journal (e.g., *IEEE Transactions on Evolutionary Computation*, *Journal of Machine Learning Research*, or *Neural Networks*), evaluating against single-policy learners (PPO, SAC) merely as "references for forgetting" rather than competitive continual-RL opponents is a fatal vulnerability. Furthermore, ENA exhibits severe performance penalties in terms of adaptation speed on CartPole and absolute return ceilings on HalfCheetah.


* **Classification of Weakness:** Primarily a combination of **insufficient state-of-the-art validation** and **methodological positioning**, compounded by performance trade-offs that limit practical applicability.

---

### 2. High-level editorial assessment

* **Originality and Novelty:** Moderate-to-High. The integration of gradient-free population-based neuroevolution, online unsupervised environment detection (via linear-Gaussian residuals and wavelet-packet chirps), and Bayesian policy reuse into a unified memory-archiving architecture (Hall of Fame) is distinct. However, individual components (PM-CuSum, BPR, MAP-Elites-style archives, triggered hypermutation) are largely drawn from existing literature.
* **Scientific Significance:** Moderate. It contributes valuable empirical evidence regarding how evolutionary neuroplasticity and memory indexing mitigate catastrophic forgetting without gradient backpropagation.
* **Relevance for the Field:** Highly relevant to continual reinforcement learning, non-stationary MDPs, and neuroevolution.
* **Likelihood of Interest for Q1/Q2 Readership:** Moderate. While researchers in evolutionary computation will find it engaging, mainstream deep reinforcement learning and continual-RL audiences will demand rigorous head-to-head comparisons against gradient-based continual-RL architectures.
* **Alignment between Objectives, Methods, Results, and Conclusions:** Good. The ablation study directly addresses the stated research questions (RQ1, RQ1a-c, RQ2). However, the conclusions regarding memory efficacy must be tempered by the realization that memory storage and test-time policy selection are inextricably coupled in the ablation design.


* **Contribution Level:** Moderate. It is an incremental-to-moderate synthesis and application of known mechanisms to non-stationary continuous control domains, lacking a unifying theoretical proof or breakthrough performance advantage.

---

### 3. Technical peer review

* **Problem Formulation & Research Question Clarity:** The problem is cleanly framed as learning across a sequence of unannounced MDPs differing in transition dynamics $P_k$. Research questions RQ1, RQ1a-c, and RQ2 are well-articulated and methodically attacked.


* **Theoretical Grounding:** The grounding borrows heavily from biological analogies (stress-induced mutation, complementary learning systems). While intuitive, these analogies do not substitute for formal theoretical guarantees regarding sample complexity, regret bounds, or stability-plasticity trade-offs.


* **Soundness of Methods:** ENA's pipeline is heavily heuristic and complex, combining linear ramp controllers, Mamdani fuzzy controllers, quadratic plasticity rules, ridge least squares dynamics models, PM-CuSum, Fixed-Share mixing, wavelet packet transforms, and Bayesian policy reuse. This high degree of hyperparameter coupling and architectural engineering raises concerns about brittleness and generalizability beyond the tuned benchmarks.


* **Appropriateness of Datasets, Benchmarks, Controls, and Comparators:** This is the manuscript's Achilles' heel. Using DQN, PPO, and SAC as mere "references" for forgetting is insufficient. Reviewers will insist on comparing ENA against established gradient-based continual-RL methods (e.g., MBCD, SWOKS, or adapted replay/regularization baselines) under identical step budgets. Claiming that such comparisons are "confounded" is unacceptable; a peer-reviewed journal paper must position itself against the empirical frontier.


* **Experimental Design & Statistical Rigor:** Excellent use of independent replicates (30 for ablations, 50 for CartPole baselines, 20 for HalfCheetah), proper power analyses, Holm correction for multiple comparisons, and clear reporting of confidence intervals and effect sizes ($d$).


* **Reproducibility and Transparency:** Outstanding. The code availability statement, fixed random seeds, detailed hyperparameter tables (Appendices B and C), and explicit step accounting represent exemplary transparency.


* **Threats to Validity:**
* *Internal Validity:* Confounding of memory storage (Hall of Fame) with test-time policy selection (Bayesian Policy Reuse vs. random baseline) in the skill ablation.


* *External Validity:* Restricted to two low-dimensional simulation domains (CartPole and HalfCheetah) with smooth, parametric physics shifts. Visual, partially observable, or contact-discontinuous environments are absent.




* **Risk of Overclaiming:** Moderate. The authors are commendably modest in their abstract and introduction ("We do not claim that ENA outperforms other continual-learning methods"), but the discussion occasionally overstates the general utility of the memory architecture given its performance deficits.



---

### 4. Section-by-section review

* **Title:**
* *What works:* Descriptive, clear, and highlights the core mechanisms.
* *What is weak/missing:* A bit long.
* *What should be revised:* Acceptable as-is, but could be tightened.


* **Abstract:**
* *What works:* Succinctly summarizes the research questions, method, ablation findings, and limitations.
* *What is weak/missing:* Acknowledges limitations upfront, which is honest, but blunts the impact.
* *What should be revised:* None.


* **Introduction:**
* *What works:* Clear contextualization of catastrophic forgetting, neuroevolution, and continual RL. Strong mapping of research questions (RQ1, RQ1a-c, RQ2).


* *What is weak/missing:* Justification for omitting direct SOTA continual-RL baselines is weak.


* *What should be revised:* Reframe the single-policy learners from "references" to baseline points of comparison, and explicitly commit to discussing or framing future work against gradient-based continual-RL frameworks.


* **Related work / literature review:**
* *What works:* Comprehensive, well-structured tabular summary (Table I) comparing ENA against EWC, CLEAR, OWL, RL-CD, MBCD, SWOKS, TeLAPA, and ITE.


* *What is weak/missing:* Lacks a deeper critical synthesis of *why* gradient-free evolution is fundamentally advantageous or disadvantageous compared to these gradient-based methods beyond empirical observation.
* *What should be revised:* Expand on the computational trade-offs of evolutionary search vs. gradient backpropagation in non-stationary optimization.


* **Methods / architecture / system design:**
* *What works:* Highly detailed algorithmic breakdown, clear operational terms, and explicit equations for Trust, Plasticity, Hall of Fame gates, and BPR.


* *What is weak/missing:* The architecture is exceptionally complex with numerous hand-tuned hyperparameters (e.g., Mamdani fuzzy rules, Fixed-Share parameters, wavelet packet parameters).
* *What should be revised:* Provide a high-level system diagram or conceptual flowchart illustrating the interaction loop between the population, detection module, and Hall of Fame.


* **Results:**
* *What works:* Thorough statistical reporting, robust replication counts, clear tables, and informative box plots.
* *What is weak/missing:* The performance penalty on HalfCheetah absolute returns is glaring.


* *What should be revised:* Ensure the text does not gloss over the fact that ENA reaches less than 20% of SAC's return on most HalfCheetah environments.


* **Discussion:**
* *What works:* Candid reflection on limitations, sample complexity of neuroevolution, and scaling issues.


* *What is weak/missing:* The discussion on why ENA's retention exceeds 100% is speculative.


* *What should be revised:* Deepen the analysis of the coupling between exploration noise and retention metrics.


* **Limitations:**
* *What works:* Explicitly lists fixed task orders, lack of training recurrence, reset requirements, and test-time selection conflation.


* *What is weak/missing:* None; this is one of the most transparent limitations sections seen in machine learning papers.
* *What should be revised:* None.


* **Conclusion:**
* *What works:* Concise wrap-up and forward-looking roadmap toward hybrid gradient-evolution architectures.


* *What is weak/missing:* None.
* *What should be revised:* None.


* **References:**
* *What works:* Broad coverage of classical and recent literature, including very recent 2026 preprints.


* *What is weak/missing:* Placeholder tags in code availability (`[TODO: release tag and archived DOI...]`).


* *What should be revised:* Ensure final DOI and Zenodo links are populated before camera-ready submission.



---

### 5. Major concerns

1. **Absence of Direct State-of-the-Art Continual-RL Baselines**
* *Issue:* The manuscript compares ENA only against unadapted single-policy learners (DQN, PPO, SAC). State-of-the-art continual-RL methods (e.g., MBCD, SWOKS) are relegated to related work with the justification that they use gradients.


* *Editorial/Scientific Importance:* In Q1 journals, demonstrating mitigation of catastrophic forgetting against models that possess no memory mechanism is insufficient. Reviewers will view this as comparing against a strawman.
* *Fix:* Implement or adapt at least one gradient-based continual RL baseline (such as an online-detected policy-switching baseline using MBCD or SWOKS dynamics models) evaluated under the same step budgets.
* *Status:* **Remediable**, but requires substantial new experimental work.


2. **Severe Absolute Performance Deficit on Continuous Control**
* *Issue:* On HalfCheetah, ENA achieves peak test returns around 2,000, whereas SAC reaches upwards of 10,000 on Mars-Slippery and 6,700 on Ice-Floor. ENA fails to reach Gymnasium's solved threshold in 0.0% of test episodes.


* *Editorial/Scientific Importance:* While ENA successfully *retains* its learned policies, the policies it learns are vastly inferior in capability to gradient-trained policies due to network size and evolutionary scaling limits.


* *Fix:* Transparently frame the contribution strictly around *structural memory retention* rather than high-performance control, or explore hybrid gradient-evolution variants as suggested in the conclusion.
* *Status:* **Remediable** via framing and discussion adjustments, though architectural scaling limitations remain inherent.


3. **Conflation of Memory Storage and Policy Selection in Ablations**
* *Issue:* The skill ablation tests variants without a Hall of Fame by deploying a random member of the final population during test, rather than using an alternative selection mechanism.


* *Editorial/Scientific Importance:* This means the ablation conflates the benefit of *archiving proven specialists* with the benefit of *intelligent test-time selection (Bayesian Policy Reuse)*.
* *Fix:* Add an ablation variant that utilizes the Hall of Fame archive combined with random selection (or naive best-Trust selection) at test time to isolate the contribution of BPR from the archive itself.
* *Status:* **Remediable** with additional evaluation runs.



---

### 6. Minor concerns

* **Placeholder Tags:** The code availability section contains unfulfilled placeholders `[TODO: release tag and archived DOI (e.g. Zenodo)...]` which must be resolved prior to final submission.


* **Notation and Terminology:** The introduction of biological metaphors ("gladiators", "Olympiads", "chirps", "niche") is extensive. While colorful, ensure precise operational definitions are always prominent to avoid semantic fatigue.


* **Figure Readability:** Figures 5, 6, and 7 are densely packed multi-panel plots; ensure axis labels and legends remain legible in single-column or double-column journal formats.



---

### 7. Hidden reviewer objections

* *Novelty Inflation:* A critical reviewer will argue that ENA is merely an assemblage of known techniques (evolutionary strategies/mutation, PM-CuSum change detection, wavelet feature extraction, and Bayesian policy reuse) stitched together with heavy heuristics. The paper must defend its synthesis more robustly.
* *Evaluation Leakage / Reset Dependency:* ENA relies heavily on environment resets during training to evaluate candidates and execute chirps. A rigorous reviewer will challenge how well this gradient-free memory mechanism scales to real-world physical systems where environment resetting is expensive, destructive, or impossible.


* *Hyperparameter Fragility:* Given the complex interplay between thresholds ($\gamma, m, \tau$), percentile windows ($H$), and fuzzy rule bases, reviewers will question the sensitivity of ENA to domain shifts outside the tested parameter ranges.

---

### 8. Journal readiness assessment

* **Readiness Level:** **Borderline Q2 / Below Q1** in its current form.
* **Top 3 Reasons for Struggle:**
1. Lack of head-to-head comparisons against modern continual-RL algorithms.
2. Substantial performance/return penalty on continuous control tasks (HalfCheetah).
3. High heuristic complexity and reliance on simulation resets.


* **Top 3 Revisions to Improve Publishability:**
1. Incorporate at least one competitive continual-RL baseline (e.g., online-detected context-switching policy memory).
2. Decouple the Hall of Fame archive ablation from Bayesian Policy Reuse to isolate memory storage vs. selection effects.
3. Streamline the methodological narrative to emphasize the principled integration of unsupervised physics detection with population-based memory storage.



---

### 9. Concrete revision roadmap

* **Critical Revisions Required for Survival:**
* Add a baseline comparison against an existing gradient-based online context-detection RL method to neutralize the "strawman baseline" objection.
* Conduct an additional ablation run testing the Hall of Fame archive paired with random/naive selection at test time to cleanly separate storage efficacy from BPR selection efficacy.


* **Important Revisions that Would Materially Strengthen the Paper:**
* Expand the discussion on environment reset assumptions and how dither-based continuous probing could replace episodic chirps for non-resettable physical systems.
* Provide sensitivity analyses on key detection parameters ($\tau, m$) to demonstrate robustness.


* **Optional Refinements:**
* Clean up remaining TODO placeholders in code availability.


* Enhance visual presentation of training dynamics and test curves in Appendix D.



---

### 10. Final recommendation to the authors

The manuscript investigates an important and challenging problem in continual reinforcement learning through an inventive, gradient-free evolutionary lens. Its transparency, thorough empirical ablation, and exceptional reproducibility standards are exemplary. However, because it avoids direct competition with state-of-the-art continual-RL baselines and incurs a steep performance deficit on continuous control benchmarks, it cannot be recommended for publication in a top-tier Q1 journal in its current state. With a major revision focused on incorporating competitive continual-RL baselines, decoupling storage from selection in the ablations, and tightening the methodological framing, this work has strong potential to meet the standards of a respected Q2 journal or an archival conference proceedings.
