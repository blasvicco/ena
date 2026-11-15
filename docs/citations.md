# Citations Verification & Audit

Last checked 2026-11-15 in four passes. Audit 4 replaced [31], whose source had no open copy, and checked the new source. Audit 3 checked [30] and [40] once their PDFs were added. Audit 1 checked what each sentence claims. Audit 2 re-checked every citing sentence in the current text, compared the metadata of all 51 bib entries with authoritative records (Crossref, arXiv, PMLR, OpenAlex, OpenReview and the PDFs' own headers), checked the cited settings of RL Baselines3 Zoo, searched again for the three missing sources, and scanned the paper for literature claims with no citation (see the last section). Page numbers ("p.") refer to the rebuilt paper `app/latex/build/main.pdf`. Source PDFs are in `references/`.

Each citation was checked with four questions:
1. **Says it?** The claim is in the cited work, at the same strength ("standard", "first", "best" need matching wording).
2. **First?** Any origin claim ("first described", "first kept") is where the idea started, or where it is widely credited as starting.
3. **Right paper?** When a sentence describes a mechanism, the cited paper contains that mechanism.
4. **Description correct?** Descriptions of other methods match what their papers report.

Status labels:
- **VERIFIED**: the source (a PDF in `references/`, or a code repository checked online) was read, and all four checks pass. A **Fixed** line records a past problem and how the paper was changed.
- **~VERIFIED**: the source was read, but at least one check fails. The **Issue** line says which one.
- **UNVERIFIED!!!**: no source is available, so nothing could be checked.

**Summary Statistics:**
- **Total Citations:** 51
- **Source checked:** 51 (100%): 50 PDFs and 1 code repository ([36])
  - **✅ VERIFIED:** 51 (15 of them fixed: on 2026-11-14, 11 after Audit 1, [8] and [51] after Audit 2, and [30] after Audit 3; on 2026-11-15, [31] after Audit 4)
  - **⚠️ ~VERIFIED:** 0
- **❌ UNVERIFIED (no source):** 0
- **Bib metadata:** all 51 entries match an authoritative record or the PDF itself, including [40], checked in Audit 3 against its title page, and [31], checked in Audit 4 against Crossref.

### Quick Status Overview:
| Range | VERIFIED | ~VERIFIED | UNVERIFIED (no source) |
|---|---|---|---|
| [1]–[10] | [1]–[10] | – | – |
| [11]–[20] | [11]–[20] | – | – |
| [21]–[30] | [21]–[30] | – | – |
| [31]–[40] | [31]–[40] | – | – |
| [41]–[51] | [41]–[51] | – | – |

---

[1] VERIFIED.
Paper says:
- Sec. I, p. 1: "When it does, for instance when a robot's physics shift between conditions, an agent with a single policy adapts by overwriting the parameters that encoded its previous behavior. This is catastrophic forgetting [1]."
- Sec. II, p. 2: "A neural network trained by gradient descent on one task after another degrades on the earlier tasks. McCloskey and Cohen named this failure catastrophic interference [1]."
- Sec. III, p. 4: "The theory was proposed in part to explain why the brain does not suffer the catastrophic interference of gradient-trained networks [1]."
Paper citation says:
```
[1] M. McCloskey and N. J. Cohen, "Catastrophic interference in connectionist networks: The sequential learning problem," in Psychology of Learning and Motivation. Academic Press, 1989, vol. 24, pp. 109--165.
```
Original content found in: `[01] McCloskey1989_CatastrophicInterference.pdf` pag. 2 (chapter p. 110)
Citation verification details: Book chapter in Psychology of Learning and Motivation, vol. 24, pp. 109–165. It shows that back-propagation networks trained sequentially (arithmetic facts; A-B/A-C lists) suffer when "new learning may interfere catastrophically with old learning". This supports the Sec. I and Sec. III sentences. "Catastrophic forgetting" is the later, now common name for the same effect.
Metadata (Audit 2): matches Crossref (DOI) on title, authors, venue, volume, issue, pages and year.
Fixed 2026-11-14 (was Q2, first): Sec. II said "This failure was first described as catastrophic interference". The chapter itself says that the disruption of old knowledge by new learning "is a recognized feature of connectionist models with distributed representations (e.g., ... Sutton, 1986)". McCloskey and Cohen named the effect ([32]: "what McCloskey and Cohen (1989) termed catastrophic interference"), so the sentence now says they named it.

---

[2] VERIFIED.
Paper says:
- Sec. I, p. 1: "Continual reinforcement learning studies agents that keep adapting without losing what they have already learned [2]."
- Sec. II, p. 2: "Continual reinforcement learning studies agents that must keep learning in such conditions without losing what they already know [2]."
- Sec. III, p. 3: "This setting is narrower than continual RL in general [2]: the observation space, the action space and the reward stay the same, and only the physics change."
Paper citation says:
```
[2] K. Khetarpal, M. Riemer, I. Rish, and D. Precup, "Towards continual reinforcement learning: A review and perspectives," Journal of Artificial Intelligence Research, vol. 75, pp. 1401--1476, 2022.
```
Original content found in: `[02] Khetarpal2022_ContinualRL.pdf` pag. 1
Citation verification details: JAIR vol. 75, pp. 1401–1476, 2022. The review defines continual RL as agents that keep learning across non-stationary tasks while they retain earlier abilities and minimize catastrophic forgetting. Its general formulation lets every part of the MDP change, so the "narrower setting" statement is consistent with it.
Metadata (Audit 2): matches Crossref (DOI) on title, authors, venue, volume, issue, pages and year; title and authors match the arXiv record.

---

[3] VERIFIED.
Paper says:
- Sec. I, p. 1: "One family of methods protects or partitions the weights of a shared network, by regularization [3], replay [4] or task-specific heads [5]."
- Sec. II, p. 2: "Elastic Weight Consolidation [3] slows learning on the weights that mattered for earlier tasks."
- Table I, p. 4: EWC, "regularized shared weights".
Paper citation says:
```
[3] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska, D. Hassabis, C. Clopath, D. Kumaran, and R. Hadsell, "Overcoming catastrophic forgetting in neural networks," Proceedings of the National Academy of Sciences, vol. 114, no. 13, pp. 3521--3526, 2017.
```
Original content found in: `[03] Kirkpatrick2017_OvercomingCatastrophicForgetting.pdf` pag. 1–2
Citation verification details: PNAS vol. 114, no. 13, pp. 3521–3526, 2017. Abstract: "Our approach remembers old tasks by selectively slowing down learning on the weights important for those tasks." Page 2 implements it as a quadratic penalty.
Metadata (Audit 2): matches Crossref (DOI) on title, authors, venue, volume, issue, pages and year; title and authors match the arXiv record.

---

[4] VERIFIED.
Paper says:
- Sec. I, p. 1: "One family of methods protects or partitions the weights of a shared network, by regularization [3], replay [4] or task-specific heads [5]."
- Sec. II, p. 2: "CLEAR [4] replays past experience with behavioral cloning, and needs no task boundaries."
- Table I, p. 4: CLEAR, "replay into shared weights".
Paper citation says:
```
[4] D. Rolnick, A. Ahuja, J. Schwarz, T. Lillicrap, and G. Wayne, "Experience replay for continual learning," in Advances in Neural Information Processing Systems, vol. 32, 2019.
```
Original content found in: `[04] Rolnick2019_ExperienceReplay.pdf` pag. 1–2
Citation verification details: NeurIPS 2019, vol. 32. Abstract: "CLEAR leverages off-policy learning and behavioral cloning from replay". Page 2: "No task information. CLEAR does not rely upon the identity of tasks or the boundaries between them."
Metadata (Audit 2): title and authors match the arXiv record. OpenAlex lists it in NeurIPS 32 with pp. 348–358, which the bib omits (optional to add).

---

[5] VERIFIED.
Paper says:
- Sec. I, p. 1: "One family of methods protects or partitions the weights of a shared network, by regularization [3], replay [4] or task-specific heads [5]."
- Sec. II, p. 2: "When tasks are incompatible, a single network cannot serve them all. For such tasks, OWL [5] gives each task its own policy head on a shared feature extractor, and chooses a head at test time with a multi-armed bandit."
- Table I, p. 4: OWL, "shared extractor (EWC), one head per task", recurrence "bandit at test".
- Sec. VI, p. 17: "OWL [5] and TeLAPA [24] are told where tasks begin and end. OWL trains the head of the task it is given, and infers the task only at test."
- Sec. VII, p. 18: "A hybrid architecture would furthermore provide a direct comparison against continual-RL frameworks designed for contextual non-stationarity: OWL [5], RL-CD [6], MBCD [7], SWOKS [8], and TeLAPA [24]."
- Sec. VII, p. 18: "OWL and TeLAPA depend on explicit task boundaries. Their authors list detecting boundaries during training [5] and weaker task boundaries [24] among future directions."
Paper citation says:
```
[5] S. Kessler, J. Parker-Holder, P. Ball, S. Zohren, and S. J. Roberts, "Same state, different task: Continual reinforcement learning without interference," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36, no. 7, 2022, pp. 7143--7151.
```
Original content found in: `[05] Kessler2022_SameEnvironment.pdf` pag. 1, 4, 7
Citation verification details: AAAI 2022, vol. 36, no. 7, pp. 7143–7151 (arXiv:2106.02940v2). Abstract: "OWL learns a factorized policy, using shared feature extraction layers, but separate heads... At test time, we formulate policy selection as a multi-armed bandit problem". The shared extractor is "regularized with EWC". Task identifiers tell OWL "when to start training on a new task", and "At test time we do not tell OWL which task it is being evaluated on". Sec. I, Sec. II, Table I and Sec. VI are correct.
Metadata (Audit 2): matches Crossref (DOI) on title, authors, venue, volume, issue, pages and year; title and authors match the arXiv record.
Fixed 2026-11-14 (was Q1, strength): Sec. VII said that the authors "identify automated task boundary detection during training... as primary challenges". The source lists it as one future direction: "There are a variety of exciting future directions for this work. For instance, it is desirable to detect task boundaries during training in addition to evaluation." The sentence now says "among future directions".

---

[6] VERIFIED.
Paper says:
- Sec. I, p. 1: "Another keeps a separate policy for each detected context and recalls it when that context returns [6]–[8]."
- Sec. II, p. 3: "In RL, RL-CD [6] learned one model of the environment per context, and recognized contexts by which model predicts best. ENA follows this idea." Also: "RL-CD is tabular, while MBCD trains its models and policies by gradient descent, and SWOKS its policies."
- Table I, p. 4: RL-CD, "model and policy per context", online detection ✓, recurrence ✓, gradient-free "tabular".
- Sec. III, p. 6: "The context-detection idea of da Silva et al. [6] is followed."
- Sec. VI, pp. 17–18: "RL-CD [6], MBCD [7] and SWOKS [8] do detect contexts online without labels, and keep a policy per context, so they address the same problem." Also: "RL-CD's models are tabular and need a discretized state space."
- Sec. VII, p. 18: "A hybrid architecture would furthermore provide a direct comparison against continual-RL frameworks designed for contextual non-stationarity: OWL [5], RL-CD [6], MBCD [7], SWOKS [8], and TeLAPA [24]."
Paper citation says:
```
[6] B. C. da Silva, E. W. Basso, A. L. C. Bazzan, and P. M. Engel, "Dealing with non-stationary environments using context detection," in Proceedings of the 23rd International Conference on Machine Learning, 2006, pp. 217--224.
```
Original content found in: `[06] daSilva2006_NonStationaryRL.pdf` pag. 1–3
Citation verification details: ICML 2006, pp. 217–224. RL-CD keeps partial models of transitions and rewards. Each model's "quality... [is] inversely proportional to its prediction error", "only the model with the highest quality is activated", and a new model (with its own policy) is created when none is good enough. The models are count-based over a discrete set of states and actions, so "tabular" and "discretized state space" are fair.
Metadata (Audit 2): matches Crossref (DOI) on title, authors, venue, volume, issue, pages and year.

---

[7] VERIFIED.
Paper says:
- Sec. I, p. 1: "Another keeps a separate policy for each detected context and recalls it when that context returns [6]–[8]."
- Sec. II, p. 3: "MBCD [7] combined an ensemble of learned probabilistic dynamics models with a high-confidence change-point detector whose false-alarm rate is bounded. It created a new policy for each new context, and reused an earlier one when a context recurred." Also: "RL-CD is tabular, while MBCD trains its models and policies by gradient descent, and SWOKS its policies."
- Table I, p. 4: MBCD, "model and policy per context", online detection ✓, recurrence ✓.
- Sec. VI, p. 17: "RL-CD [6], MBCD [7] and SWOKS [8] do detect contexts online without labels, and keep a policy per context, so they address the same problem."
- Sec. VII, p. 18: "A hybrid architecture would furthermore provide a direct comparison against continual-RL frameworks designed for contextual non-stationarity: OWL [5], RL-CD [6], MBCD [7], SWOKS [8], and TeLAPA [24]."
Paper citation says:
```
[7] L. N. Alegre, A. L. C. Bazzan, and B. C. da Silva, "Minimum-delay adaptation in non-stationary reinforcement learning via online high-confidence change-point detection," in Proceedings of the 20th International Conference on Autonomous Agents and Multiagent Systems (AAMAS), 2021, pp. 97--105.
```
Original content found in: `[07] Alegre2021_MinimumWidth.pdf` pag. 1–4
Citation verification details: AAMAS 2021, pp. 97–105 (arXiv:2105.09452). Abstract: the method "bounds the rate of false alarm" and builds "a mixture model composed of a (possibly infinite) ensemble of probabilistic dynamics predictors". Each context has a "bootstrap ensemble of probabilistic neural networks", detection uses MCUSUM, and policies are trained with SAC. New policies are created for new contexts and earlier ones are reused.
Metadata (Audit 2): title and authors match the arXiv record; the arXiv journal reference confirms AAMAS 2021, pp. 97–105.

---

[8] VERIFIED.
Paper says:
- Sec. I, p. 1: "Another keeps a separate policy for each detected context and recalls it when that context returns [6]–[8]."
- Sec. II, p. 3: "SWOKS [8] detects contexts with optimal-transport distances and Kolmogorov–Smirnov tests, and trains one policy per context." Also: "RL-CD is tabular, while MBCD trains its models and policies by gradient descent, and SWOKS its policies."
- Table I, p. 4: SWOKS, "policy per context", online detection ✓, recurrence ✓.
- Sec. VI, p. 17: "RL-CD [6], MBCD [7] and SWOKS [8] do detect contexts online without labels, and keep a policy per context, so they address the same problem."
- Sec. VII, p. 18: "A hybrid architecture would furthermore provide a direct comparison against continual-RL frameworks designed for contextual non-stationarity: OWL [5], RL-CD [6], MBCD [7], SWOKS [8], and TeLAPA [24]."
Paper citation says:
```
[8] J. Dick, S. Nath, C. Peridis, E. Benjamin, S. Kolouri, and A. Soltoggio, "Statistical context detection for deep lifelong reinforcement learning," in Proceedings of the 3rd Conference on Lifelong Learning Agents (CoLLAs), 2024.
```
Original content found in: `[08] Dick2024_Statistical.pdf` pag. 1, 6
Citation verification details: CoLLAs 2024. SWOKS (Sliced Wasserstein Online Kolmogorov-Smirnov) measures "distances... obtained via optimal transport methods, i.e., Wasserstein distance" and applies "an adapted Kolmogorov-Smirnov calculation". Each task's policy is a modulating mask on a shared backbone, trained with PPO, or with SAC on Half-Cheetah. That is one policy per context, trained by gradient descent.
Metadata (Audit 2): venue confirmed by the PDF header ("Published at 3rd Conference on Lifelong Learning Agents (CoLLAs), 2024").

Fixed 2026-11-14 (Audit 2, was Q4, description): Sec. II said "MBCD and SWOKS train their models and policies by gradient descent". SWOKS keeps no environment model: it detects contexts from latent action-reward embeddings of the policy network and trains only policies. The sentence now says "MBCD trains its models and policies by gradient descent, and SWOKS its policies".
---

[9] VERIFIED.
Paper says:
- Sec. I, p. 1: "Nisioti et al. [9] showed that evolution strategies balance adapting to new tasks and remembering old ones well, while genetic algorithms learn new tasks best but forget more. In these methods, however, the whole population evolves toward the current task. The population receives no signal that the environment has changed..."
- Sec. II, p. 2: "Nisioti et al. [9] compared evolution strategies and genetic algorithms with PPO, continual RL methods and population-based RL, under changes of task and environment. Evolution strategies achieved the best balance between adapting and remembering. Genetic algorithms were the most plastic, but forgot more. The authors related this balance to how much the neighborhoods of good solutions in weight space overlap between consecutive tasks."
- Table I, p. 4: Nisioti et al., "population only", gradient-free ✓.
Paper citation says:
```
[9] E. Nisioti, A. Cossu, K. Korte, and S. Risi, "Continual reinforcement learning with neuroevolution," arXiv preprint arXiv:2610.01583, 2026.
```
Original content found in: `[09] Nisioti2026_ContinualQD.pdf` pag. 1–3, 5
Citation verification details: arXiv:2610.01583v1. The paper compares ES and a GA with PPO, three continual-RL variants and population-based RL. It has no quality-diversity archive and no memory (the previous note here was wrong about that). "ES achieves the best stability-plasticity trade-off. The GA is the most plastic method in almost every environment". "The size of the overlap between the neighborhoods of consecutive tasks correlates with a method's stability-plasticity trade-off". The setting is boundary-free: "no information about task change is provided". Sec. II and Table I are correct.
Metadata (Audit 2): title and authors match the arXiv record.
Fixed 2026-11-14 (was Q4, description): Sec. I said genetic algorithms "adapt fastest", a speed claim. Nisioti et al. measure plasticity as learning accuracy, "the return on each task at the end of training", and write that "the GA learn[s] new tasks better than ES but forgets more". The sentence now says "learn new tasks best". It avoids "plastic" because the introduction uses Plasticity for ENA's search effort.

---

[10] VERIFIED.
Paper says:
- Sec. I, p. 2: "Specialist selection governs which archived specialist to deploy via Bayesian policy reuse [10], integrating detector evidence with observed episodic returns."
- Sec. II, p. 3: "To choose which stored policy to deploy, Bayesian Policy Reuse [10] keeps a belief over task types. The belief is updated from a signal of each episode, such as its return or its transitions. ENA adapts it to an online, non-stationary setting, and updates the belief from the returns and the detector's likelihoods together (Section III-I)."
- Sec. III, p. 7: "This choice is made with Bayesian Policy Reuse (BPR) [10]. ENA's BPR combines two kinds of evidence in one posterior: the dynamics evidence Lj and the observed utility u." Also, p. 8: "BPR builds its performance model offline, by playing every policy in every type." and "BPR assumes one fixed type per trial."
Paper citation says:
```
[10] B. Rosman, M. Hawasly, and S. Ramamoorthy, "Bayesian policy reuse," Machine Learning, vol. 104, no. 1, pp. 99--127, 2016.
```
Original content found in: `[10] Rosman2016_BayesianPolicyReuse.pdf` pag. 7–11
Citation verification details: Machine Learning vol. 104, 2016 (arXiv:1505.00284v2). BPR keeps a type belief updated by Bayes' rule from an observation signal (Eq. 1–2). The observation models are learned offline: "The agent runs all the policies from the library Π on all the instances of τ". It responds to a single "new task instance". The offline model and the fixed type are described correctly.
Metadata (Audit 2): matches Crossref (DOI) on title, authors, venue, volume, issue, pages and year; title and authors match the arXiv record.
Fixed 2026-11-14 (was Q4, description): Sec. II said BPR's belief is "updated from the returns it observes" and that ENA "adds the dynamics models' likelihoods". Sec. III credited BPR itself with combining both kinds of evidence. Rosman et al. leave the signal open, "the form of this signal remains unspecified" (§2.8), and list state-action-state tuples, rewards and episodic returns as examples. Sec. II now names returns and transitions as possible signals and says ENA uses the returns and the detector's likelihoods together. Sec. III now says "ENA's BPR".

---

[11] VERIFIED.
Paper says:
- Sec. I, p. 2: "In CartPole [11], the agent balances a pole on a moving cart."
- Sec. IV, p. 8: "ENA was evaluated in two non-stationary domains built on Gymnasium [46]: the discrete-action CartPole-v1 [11], and the continuous-control MuJoCo [12] HalfCheetah-v5."
Paper citation says:
```
[11] A. G. Barto, R. S. Sutton, and C. W. Anderson, "Neuronlike adaptive elements that can solve difficult learning control problems," IEEE Transactions on Systems, Man, and Cybernetics, vol. SMC-13, no. 5, pp. 834--846, 1983.
```
Original content found in: `[11] Barto1983_NeuronLike.pdf` pag. 2–3 (scanned; journal pp. 835–836)
Citation verification details: IEEE Trans. SMC, vol. SMC-13, no. 5, pp. 834–846, 1983. The scanned article uses "the pole-balancing control problem posed by Michie and Chambers" as its test bed. This is the classic source of the cart-pole task that Gymnasium's CartPole implements.
Metadata (Audit 2): matches Crossref (DOI) on title, authors, venue, volume, issue, pages and year.

---

[12] VERIFIED.
Paper says:
- Sec. I, p. 2: "In HalfCheetah, a two-legged robot simulated in MuJoCo [12] learns to run."
- Sec. IV, p. 8: "ENA was evaluated in two non-stationary domains built on Gymnasium [46]: the discrete-action CartPole-v1 [11], and the continuous-control MuJoCo [12] HalfCheetah-v5."
Paper citation says:
```
[12] E. Todorov, T. Erez, and Y. Tassa, "{MuJoCo}: A physics engine for model-based control," in 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, 2012, pp. 5026--5033.
```
Original content found in: `[12] Todorov2012_MuJoCo.pdf` pag. 1
Citation verification details: IEEE/RSJ IROS 2012, pp. 5026–5033. The paper introduces the MuJoCo physics engine. Citing it for the simulator (Sec. IV) is correct.
Metadata (Audit 2): matches Crossref (DOI) on title, authors, venue, volume, issue, pages and year.
Fixed 2026-11-14 (was Q3, right paper): Sec. I said "In HalfCheetah [12]...", attaching the citation to the task. The MuJoCo paper contains no cheetah; the task comes from Wawrzyński (2009). The sentence now cites [12] for the simulator only. Crediting the task itself (Wawrzyński 2009) was not chosen.

---

[13] VERIFIED.
Paper says:
- Sec. I, p. 2: "Alongside it, standard single-policy learners were trained on the same sequences of environments, with the same number of environment steps: DQN [13], PPO [14] and SAC [15]."
- Sec. IV, p. 9: "On CartPole, ENA-Combined was compared with DQN [13], PPO [14] and the discrete-action variant of SAC [47]."
Paper citation says:
```
[13] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis, "Human-level control through deep reinforcement learning," Nature, vol. 518, no. 7540, pp. 529--533, 2015.
```
Original content found in: `[13] Mnih2015_HumanLevelDQN.pdf` pag. 1
Citation verification details: Nature vol. 518, no. 7540, pp. 529–533, 2015 (doi:10.1038/nature14236). It introduces the deep Q-network (DQN).
Metadata (Audit 2): matches Crossref (DOI) on title, authors, venue, volume, issue, pages and year.

---

[14] VERIFIED.
Paper says:
- Sec. I, p. 2: "Alongside it, standard single-policy learners were trained on the same sequences of environments, with the same number of environment steps: DQN [13], PPO [14] and SAC [15]."
- Sec. IV, p. 9: "On CartPole, ENA-Combined was compared with DQN [13], PPO [14] and the discrete-action variant of SAC [47]."
Paper citation says:
```
[14] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," arXiv preprint arXiv:1707.06347, 2017.
```
Original content found in: `[14] Schulman2017_PPO.pdf` pag. 3
Citation verification details: arXiv:1707.06347, 2017. Section 3 introduces PPO and its clipped surrogate objective.
Metadata (Audit 2): title and authors match the arXiv record.

---

[15] VERIFIED.
Paper says:
- Sec. I, p. 2: "Alongside it, standard single-policy learners were trained on the same sequences of environments, with the same number of environment steps: DQN [13], PPO [14] and SAC [15]."
- Sec. IV, p. 9: "On HalfCheetah, it was compared with PPO and SAC [15]."
Paper citation says:
```
[15] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor," in Proceedings of the 35th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 80, 2018, pp. 1861--1870.
```
Original content found in: `[15] Haarnoja2018_SAC.pdf` pag. 2–5
Citation verification details: ICML 2018 (PMLR vol. 80), pp. 1861–1870. Sections 3–4 formulate Soft Actor-Critic.
Metadata (Audit 2): title and authors match the arXiv record; the PMLR page confirms PMLR 80:1861–1870.

---

[16] VERIFIED.
Paper says:
- Sec. II, p. 2: "Neuroevolution optimizes neural networks by mutation and selection over a population, instead of by gradients [16]."
Paper citation says:
```
[16] K. O. Stanley, J. Clune, J. Lehman, and R. Miikkulainen, "Designing neural networks through neuroevolution," Nature Machine Intelligence, vol. 1, no. 1, pp. 24--35, 2019.
```
Original content found in: `[16] Stanley2019_DesigningNeuralNetworks.pdf` pag. 1–2
Citation verification details: Nature Machine Intelligence vol. 1, pp. 24–35, 2019 (doi:10.1038/s42256-018-0006-z). Abstract: neuroevolution "harnesses evolutionary algorithms to optimize neural networks" and is "an alternative approach" to "variants of stochastic gradient descent". It works by "maintaining a population of solutions during search", and "through mutation and crossover, the population gradually evolves".
Metadata (Audit 2): matches Crossref (DOI) on title, authors, venue, volume, issue, pages and year.

---

[17] VERIFIED.
Paper says:
- Sec. II, p. 2: "Evolution strategies [17] and genetic algorithms [18] are competitive with gradient-based RL on standard benchmarks."
- Sec. III, pp. 4–5: "On HalfCheetah, the hidden layers have 32 × 16 tanh units. Tanh hidden layers follow the evolution-strategies policy of Salimans et al. [17], and normc initialization follows its released code [36]. The output is also a tanh there, as in the actor networks of CEM-RL [20], which keeps actions in their range."
Paper citation says:
```
[17] T. Salimans, J. Ho, X. Chen, S. Sidor, and I. Sutskever, "Evolution strategies as a scalable alternative to reinforcement learning," arXiv preprint arXiv:1703.03864, 2017.
```
Original content found in: `[17] Salimans2017_EvolutionStrategies.pdf` pag. 1–2, 7
Citation verification details: arXiv:1703.03864, 2017. ES "obtain[s] competitive results on most Atari games" and matches TRPO on MuJoCo "using no more than 10x as much data", so the Sec. II sentence holds. The MuJoCo policies are "multilayer perceptrons with two 64-unit hidden layers separated by tanh nonlinearities".
Metadata (Audit 2): title and authors match the arXiv record.
Fixed 2026-11-14 (was Q3, right paper): Sec. III said "normc initialization and a tanh output, as in the evolution-strategies policy of Salimans et al. [17]". The paper only states tanh hidden layers. normc comes from its released code [36], whose continuous output is a linear layer with no tanh. The sentence now credits [17] for the tanh hidden layers, [36] for normc, and CEM-RL [20] (Appendix A, Table 3) for the tanh output. The comment in `app/src/half_cheetah.py` (`ena_brain`) was corrected the same way.

---

[18] VERIFIED.
Paper says:
- Sec. II, p. 2: "Evolution strategies [17] and genetic algorithms [18] are competitive with gradient-based RL on standard benchmarks."
Paper citation says:
```
[18] F. P. Such, V. Madhavan, E. Conti, J. Lehman, K. O. Stanley, and J. Clune, "Deep neuroevolution: Genetic algorithms are a competitive alternative for training deep neural networks for reinforcement learning," arXiv preprint arXiv:1712.06567, 2017.
```
Original content found in: `[18] Such2017_DeepNeuroevolution.pdf` pag. 1
Citation verification details: arXiv:1712.06567. The title states that GAs "are a competitive alternative". The abstract says a gradient-free GA "performs well on hard deep RL problems, including Atari and humanoid locomotion".
Metadata (Audit 2): title and authors match the arXiv record.

---

[19] VERIFIED.
Paper says:
- Sec. II, p. 2: "They need many samples, however [19], [20], and optimizing many parameters remains hard for them [19]." Also: "Hybrid methods use a population to diversify the experience of a gradient-trained learner [19]."
- Sec. VI, p. 17: "Evolutionary algorithms need many samples [19], [20], and struggle on problems that require optimizing many parameters [19]."
- Sec. VII, p. 18: "Both limitations align with known properties of evolutionary search, which exhibits high sample complexity and scales unfavorably with parameter count [19], [49]." Also: "The evolutionary population could also supply diverse rollouts to off-policy replay buffers, as in evolutionary reinforcement learning [19]."
Paper citation says:
```
[19] S. Khadka and K. Tumer, "Evolution-guided policy gradient in reinforcement learning," in Advances in Neural Information Processing Systems, vol. 31, 2018, pp. 1188--1200.
```
Original content found in: `[19] Khadka2018_EvolutionGuidedPolicyGradient.pdf` pag. 1–2
Citation verification details: NeurIPS 2018, vol. 31. Abstract: "EAs typically suffer from high sample complexity and struggle to solve problems that require optimization of a large number of parameters." ERL "uses the population of an EA to provide diversified data to train an RL agent" through a replay buffer. Every sentence is supported, almost word for word.
Metadata (Audit 2): title and authors match the arXiv record; OpenAlex confirms NeurIPS 31, pp. 1188–1200.

---

[20] VERIFIED.
Paper says:
- Sec. II, p. 2: "They need many samples, however [19], [20], and optimizing many parameters remains hard for them [19]."
- Sec. III, pp. 4–5: "The output is also a tanh there, as in the actor networks of CEM-RL [20], which keeps actions in their range." ("there" is HalfCheetah; CartPole's output layer is linear.)
- Sec. VI, p. 17: "Evolutionary algorithms need many samples [19], [20], and struggle on problems that require optimizing many parameters [19]."
Paper citation says:
```
[20] A. Pourchot and O. Sigaud, "{CEM-RL}: Combining evolutionary and gradient-based methods for policy search," in International Conference on Learning Representations (ICLR), 2019.
```
Original content found in: `[20] Pourchot2019_CEMRL.pdf` pag. 1–2
Citation verification details: ICLR 2019 (arXiv:1810.01222). Supports the first half: "Evolutionary methods are significantly less sample efficient than deep RL methods because they learn from complete episodes". Appendix A, Table 3 (the actor has tanh hidden layers and a tanh output; tested on half-cheetah-v2) is the source for ENA's tanh output (Sec. III, p. 4).
Metadata (Audit 2): title and authors match the arXiv record; OpenReview and DBLP confirm ICLR 2019 (poster).
Fixed 2026-11-14 (was Q1, says it): Sec. II and Sec. VI cited [20] for both "many samples" and "many parameters". CEM-RL supports only the first ("significantly less sample efficient"); on parameter count it only hedges ("might deteriorate the estimation of the covariance matrices"). [20] now supports only the sample clause, and the parameter clause cites [19] alone.

---

[21] VERIFIED.
Paper says:
- Sec. II, p. 2: "Earlier work on evolutionary algorithms for changing optimization problems developed remedies such as increasing diversity after a change, maintaining diversity throughout the run, and memory [21]."
Paper citation says:
```
[21] D. Yazdani, R. Cheng, D. Yazdani, J. Branke, Y. Jin, and X. Yao, "A survey of evolutionary continuous dynamic optimization over two decades---part {A}," IEEE Transactions on Evolutionary Computation, vol. 25, no. 4, pp. 609--629, 2021.
```
Original content found in: `[21] Yazdani2021_SurveyDynamicOptimization.pdf` pag. 2
Citation verification details: IEEE TEVC vol. 25, no. 4, pp. 609–629, 2021. The survey reports the classic taxonomy of components for dynamic optimization: "generating diversity after a change, maintaining diversity over time, using multi-population, and memory based approaches". The taxonomy comes from Jin & Branke (2005), which the paper does not cite.
Metadata (Audit 2): matches Crossref (DOI) on title, authors, venue, volume, issue, pages and year.

---

[22] VERIFIED.
Paper says:
- Sec. II, p. 2: "Quality-Diversity algorithms such as MAP-Elites [22] search for many diverse, high-performing solutions instead of a single optimum, and store them in an archive."
Paper citation says:
```
[22] J.-B. Mouret and J. Clune, "Illuminating search spaces by mapping elites," arXiv preprint arXiv:1504.04909, 2015.
```
Original content found in: `[22] Mouret2015_IlluminatingSearch.pdf` pag. 1–3
Citation verification details: arXiv:1504.04909. MAP-Elites keeps the highest-performing solution found in each cell of a feature-space map, instead of a single optimum.
Metadata (Audit 2): title and authors match the arXiv record.

---

[23] VERIFIED.
Paper says:
- Sec. II, pp. 2–3: "Intelligent Trial and Error (ITE) [23] computes such an archive offline, in simulation. With it, a damaged robot finds a working gait within minutes, by trying archived behaviors guided by a Bayesian model."
- Table I, p. 4: ITE, "offline behavior archive", online detection "performance drop", gradient-free ✓, probing "trials".
- Sec. VI, p. 17: "ITE [23] builds its archive offline, before deployment. It adapts once to each damage, starting when performance drops below a user-set threshold. It neither keeps a memory across a sequence of conditions, nor recognizes a condition that returns."
- Sec. VII, pp. 18–19: "ITE initiates adaptation based on a predefined performance threshold and maintains no memory across sequential conditions [23]."
Paper citation says:
```
[23] A. Cully, J. Clune, D. Tarapore, and J.-B. Mouret, "Robots that can adapt like animals," Nature, vol. 521, no. 7553, pp. 503--507, 2015.
```
Original content found in: `[23] Cully2015_RobotsAdapt.pdf` pag. 1–2, 16
Citation verification details: Nature vol. 521, no. 7553, pp. 503–507, 2015 (arXiv:1407.3501). The behavior-performance map is created in simulation. Robots "adapt to damage in less than two minutes" through a map-based Bayesian optimization with a Gaussian process. Methods: "The adaptation step is triggered when the performance drops by a certain amount. The simplest way to choose that threshold is to let the user specify it." The "no memory across conditions" statement is a reading of the method: each adaptation starts from the simulated map, and the paper evaluates one damage at a time. It is correct, but the paper never states it.
Metadata (Audit 2): matches Crossref (DOI) on title, authors, venue, volume, issue, pages and year; title and authors match the arXiv record.

---

[24] VERIFIED.
Paper says:
- Sec. II, p. 3: "TeLAPA [24] brings the same idea to continual RL. For each task, it trains a base policy with PPO and fills an archive of behaviorally diverse policies around it with MAP-Elites. When a new or revisited task begins, short adaptation probes pick a starting point from all archives, so that competent alternatives remain available when a task is revisited. TeLAPA is evaluated on MiniGrid and on cyclic friction shifts in MuJoCo, HalfCheetah included, with the task boundaries given." Also, p. 3: "TeLAPA is partly gradient-free: it trains with PPO, and fills its archives by mutation."
- Table I, p. 4: TeLAPA, "per-task archives of initializations", online detection –, recurrence "by label", gradient-free "partly", probing "few-shot probes".
- Sec. VI, p. 17: "OWL [5] and TeLAPA [24] are told where tasks begin and end... TeLAPA builds a task's archive, and retrieves a starting point from its archives, at each task boundary it is given."
- Sec. VII, p. 18: "A hybrid architecture would furthermore provide a direct comparison against continual-RL frameworks designed for contextual non-stationarity: OWL [5], RL-CD [6], MBCD [7], SWOKS [8], and TeLAPA [24]."
- Sec. VII, p. 18: "OWL and TeLAPA depend on explicit task boundaries. Their authors list detecting boundaries during training [5] and weaker task boundaries [24] among future directions."
Paper citation says:
```
[24] L. Lillo and N. Cheney, "Beyond single-model optimization: Preserving plasticity in continual reinforcement learning," in Proceedings of the 5th Conference on Lifelong Learning Agents (CoLLAs), 2026, arXiv:2604.15414.
```
Original content found in: `[24] Lillo2026_BeyondSingleTask.pdf` pag. 1–5, Appendix D and the MuJoCo appendix
Citation verification details: CoLLAs 2026 (arXiv:2604.15414v3). TeLAPA stands for "Transfer-Enabled Latent-Aligned Policy Archives" (the previous note gave a wrong expansion). It "trains a base policy with PPO and populates an archive of diverse elites around it using parameter-space MAP-Elites". "Candidate initializations are retrieved from prior archives... and then selected through short transfer-time evaluations". Algorithm 2 receives the task sequence with its labels, and `BaseTag(τt)` maps a revisit to its original archive. An appendix adds cyclic friction shifts on HalfCheetah-v5, Hopper-v5 and Walker2d-v5. Sec. II, Sec. VI and Table I are correct.
Metadata (Audit 2): title and authors match the arXiv record; venue confirmed by the PDF header (CoLLAs 2026).
Fixed 2026-11-14 (was Q1, strength): Sec. VII said the authors identify "handling ambiguous transitions" as a primary challenge. The source lists "weaker task boundaries" as one item of a future-work list ("longer task streams, weaker task boundaries, richer visual domains, continuous-control environments..."). The sentence now uses the source's words and says "among future directions".
Fixed 2026-11-15 (Table I, was an internal contradiction): Table I gave TeLAPA "–" for recurrence, although the text says it serves revisited tasks, using the labels it is given (Algorithm 2, `BaseTag`). The cell now reads "by label".

---

[25] VERIFIED.
Paper says:
- Sec. II, p. 3: "Classic methods range from Page's CUSUM [25] to Bayesian online changepoint detection [26]."
- Sec. III, p. 7: "The monitor is the Predictive-Mixture CuSum (PM-CuSum) of Halme, Poor and Koivunen [27], a recent successor of Page's CUSUM [25]."
Paper citation says:
```
[25] E. S. Page, "Continuous inspection schemes," Biometrika, vol. 41, no. 1--2, pp. 100--115, 1954.
```
Original content found in: `[25] Page1954_ContinuousSampling.pdf` pag. 1–5
Citation verification details: Biometrika vol. 41, no. 1/2, pp. 100–115, 1954. Page introduces the cumulative-score scheme and its decision rule "S_n − min S_i ≥ h" for detecting a change in a parameter. Page 1954 is the accepted origin of CUSUM ([27] p. 1: "the Cumulative Sum (CuSum) procedure introduced by Page").
Metadata (Audit 2): matches Crossref (DOI) on title, authors, venue, volume, issue, pages and year.

---

[26] VERIFIED.
Paper says:
- Sec. II, p. 3: "Classic methods range from Page's CUSUM [25] to Bayesian online changepoint detection [26]."
Paper citation says:
```
[26] R. P. Adams and D. J. C. MacKay, "Bayesian online changepoint detection," arXiv preprint arXiv:0710.3742, 2007.
```
Original content found in: `[26] Adams2007_BayesianChangepoint.pdf` pag. 1–2
Citation verification details: arXiv:0710.3742. Exact online Bayesian changepoint detection by message passing over the run length.
Metadata (Audit 2): title and authors match the arXiv record.

---

[27] VERIFIED.
Paper says:
- Sec. II, p. 3: "On CartPole, ENA uses the Predictive-Mixture CuSum [27]. It detects a change away from a known distribution without a model of the distribution after the change, by mixing predictors fitted on recent windows of data."
- Sec. III, p. 7: "The monitor is the Predictive-Mixture CuSum (PM-CuSum) of Halme, Poor and Koivunen [27], a recent successor of Page's CUSUM [25]." Also: "For an exact pre-change distribution, Halme et al. show that this threshold keeps the average run length above γ."
- Sec. III, p. 7: "A departure from the Gaussian example of [27] is made in one respect. Each window also estimates a variance..."
Paper citation says:
```
[27] T. Halme, H. V. Poor, and V. Koivunen, "Adaptive sequential change detection using mixtures of predictive distributions," arXiv preprint arXiv:2606.05072, 2026.
```
Original content found in: `[27] Halme2026_SequentialChangepoint.pdf` pag. 1, 5–7, Sec. 4.2
Citation verification details: arXiv:2606.05072v3. PM-CuSum "combines predictive distributions constructed from sliding windows of different lengths within a CuSum recursion", for an unknown post-change distribution. The weights use "the Fixed Share (FS) algorithm". Lemma 3.1: "setting b = log γ is sufficient to guarantee ARL(T_b) ≥ γ". The Gaussian example (Sec. 4.2) detects mean shifts with a known variance, so a per-window variance is a real departure.
Metadata (Audit 2): title and authors match the arXiv record (the bib title is the current arXiv title; only the heading inside the local v3 PDF is shorter).

---

[28] VERIFIED.
Paper says:
- Sec. III, p. 4: "Bacteria under stress increase their mutation rate. The response is regulated and transient. Variation rises when the current genotype is failing and stays low otherwise, because most mutations in a well-adapted organism are harmful or neutral [28]."
Paper citation says:
```
[28] R. S. Galhardo, P. J. Hastings, and S. M. Rosenberg, "Mutation as a stress response and the regulation of evolvability," Critical Reviews in Biochemistry and Molecular Biology, vol. 42, no. 5, pp. 399--435, 2007.
```
Original content found in: `[28] Galhardo2007_MutationStressResponse.pdf` pag. 1 (p. 399), 3 (p. 401)
Citation verification details: Crit. Rev. Biochem. Mol. Biol. vol. 42, no. 5, pp. 399–435, 2007 (doi:10.1080/10409230701648502). The review describes "the regulation of mutagenesis in time by cellular stress responses, which promote random mutations specifically when cells are poorly adapted to their environments, i.e., when they are stressed". Strains "increase mutation rate transiently during stress". This supports "regulated", "transient" and "rises when failing".
Metadata (Audit 2): matches Crossref (DOI) on title, authors, venue, volume, issue, pages and year.
Fixed 2026-11-14 (was Q1, strength): the sentence said "most mutations harm a well-adapted organism". The source says "Most mutations in well adapted organisms are deleterious or neutral". The sentence now says "harmful or neutral".

---

[29] VERIFIED.
Paper says:
- Sec. III, p. 4: "Evolutionary computation adopted the idea as triggered hypermutation, which raises the mutation rate when performance drops in a non-stationary environment [29]."
Paper citation says:
```
[29] H. G. Cobb, "An investigation into the use of hypermutation as an adaptive operator in genetic algorithms having continuous, time-dependent nonstationary environments," Naval Research Laboratory, Washington, DC, Tech. Rep. AIC-90-001, 1990.
```
Original content found in: `[29] Cobb1990_Investigation.pdf` pag. 2, 5
Citation verification details: The PDF is NRL Memorandum Report 6760 (DTIC AD-A229 159), December 1990. The bib's "AIC-90-001" is the NCARAI number of the same report; the PDF shows only MR 6760. Abstract: "During periods where the time-averaged best performance of the GA worsens, the GA enters hypermutation (a large increase in mutation)". Cobb draws the idea from biology: "when cells are stressed by environmental conditions, some of the cells tend to enter a 'hypermutable' state". So "adopted the idea" holds, and Cobb 1990 is the credited origin. Note: the report says "hypermutation" or "adaptive mutation operator". The name "triggered hypermutation" appears later (Cobb & Grefenstette 1993). The previous note here quoted it as if it were in this report.
Metadata (Audit 2): the PDF is NRL Memorandum Report 6760; the bib gives the same report by its NCARAI number, AIC-90-001 (optional: give both).

---

[30] VERIFIED.
Paper says:
- Sec. III, p. 4: "More generally, it studies the control of an algorithm's parameters during the run [30]."
Paper citation says:
```
[30] G. Karafotias, M. Hoogendoorn, and A. E. Eiben, "Parameter control in evolutionary algorithms: Trends and challenges," IEEE Transactions on Evolutionary Computation, vol. 19, no. 2, pp. 167--187, 2015.
```
Original content found in: `[30] Karafotias2015_ParameterControl.pdf` pag. 1, 3, 15–16 (accepted manuscript, IEEE early-access version)
Citation verification details: IEEE TEVC vol. 19, no. 2, pp. 167–187, 2015 (doi:10.1109/TEVC.2014.2308294). The PDF is the accepted manuscript ("accepted for publication in a future issue... Content may change prior to final publication"), so its page numbers (1–22) differ from the journal's. A survey of parameter control in evolutionary algorithms. It defines control as the method that "changes the values while the EA is running" (p. 3), as opposed to tuning, where values "remain fixed during the run". Among the advantages, control "allows the EA to adjust to the changing fitness landscapes when facing dynamic problems" (p. 1). So it is the right source for "controlling an algorithm's parameters during the run".
Metadata (Audit 2): matches Crossref (DOI) on title, authors, venue, volume, issue, pages and year. Confirmed by the PDF on title, authors and DOI.
Fixed 2026-11-14 (Audit 3, was Q1, strength): the sentence said "More generally, it adopted the idea of controlling an algorithm's parameters during the run". The survey's conclusion is that "the evolutionary computing community did not adopt parameter control techniques as part of the standard machinery" (p. 16), and that "no parameter control method or strategy has been widely adopted by the community" (p. 15). The sentence now says evolutionary computation "studies the control of an algorithm's parameters during the run".
---

[31] VERIFIED.
Paper says:
- Sec. III, p. 4: "Freshly initialized specialists also enter the population in every generation, as random immigrants [31] do. Their purpose is to keep the population diverse enough to follow a moving optimum."
Paper citation says:
```
[31] R. Tin{\'o}s and S. Yang, "A self-organizing random immigrants genetic algorithm for dynamic optimization problems," Genetic Programming and Evolvable Machines, vol. 8, no. 3, pp. 255--286, 2007.
```
Original content found in: `[31] Tinos2007_SelfOrganizingRandomImmigrants.pdf` pag. 2–4 (pp. 256–258)
Citation verification details: Genetic Programming and Evolvable Machines vol. 8, no. 3, pp. 255–286, 2007 (doi:10.1007/s10710-007-9024-z). The PDF is the published Springer version. Sec. 2 describes the standard random immigrants GA, with pseudocode (Algorithm 1), before the paper introduces its self-organizing variant. "In a genetic algorithm (GA) with random immigrants, a fraction of the current population is replaced by randomly generated individuals in each generation of the run" (p. 256). "The random immigrants approach tries to maintain the diversity level of the population, which can be very useful to prepare the GA for possible changes in the fitness landscape" (p. 256). Both sentences of the paper are supported. The paper credits the scheme to Grefenstette: "The Random Immigrants GA (RIGA) proposed by Grefenstette [13]" (p. 257). Its ref. 13 is Grefenstette, PPSN 2, pp. 137–144, 1992.
Metadata (Audit 4): matches Crossref (DOI) on title, authors, venue, volume, issue, pages and year. Confirmed by the PDF's header.
Fixed 2026-11-15 (Audit 4, was UNVERIFIED, no source): [31] was Grefenstette, "Genetic algorithms for changing environments," PPSN 2, 1992, a paywalled book chapter with no open copy. The sentence makes no origin claim; it only needs the mechanism. So [31] now cites Tinós and Yang 2007, which describes that mechanism and credits Grefenstette 1992 for it. Not chosen: Yang 2008 (Evolutionary Computation 16(3):385–416, doi:10.1162/evco.2008.16.3.385), which also supports the sentence (p. 388), and Cobb and Grefenstette 1993 (ICGA 1993, pp. 523–530; DTIC report doi:10.21236/ADA294075), which could not be downloaded.

---

[32] VERIFIED.
Paper says:
- Sec. III, p. 4: "The complementary learning systems theory [32], [33] holds that the brain learns quickly in the hippocampus and consolidates slowly into the neocortex. There, knowledge is protected from interference by new learning."
Paper citation says:
```
[32] J. L. McClelland, B. L. McNaughton, and R. C. O'Reilly, "Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and memory," Psychological Review, vol. 102, no. 3, pp. 419--457, 1995.
```
Original content found in: `[32] McClelland1995_ComplementaryLearning.pdf` pag. 1 (p. 419), catastrophic-interference section
Citation verification details: Psychological Review vol. 102, no. 3, pp. 419–457, 1995. Abstract: "the neocortex learns slowly to discover the structure in ensembles of experiences. The hippocampal system permits rapid learning of new items without disrupting this structure". The paper builds its argument on "what McCloskey and Cohen (1989) termed catastrophic interference", which supports the Sec. III sentence citing [1].
Metadata (Audit 2): matches Crossref (DOI) on title, authors, venue, volume, issue, pages and year.

---

[33] VERIFIED.
Paper says:
- Sec. III, p. 4: "The complementary learning systems theory [32], [33] holds that the brain learns quickly in the hippocampus and consolidates slowly into the neocortex."
Paper citation says:
```
[33] D. Kumaran, D. Hassabis, and J. L. McClelland, "What learning systems do intelligent agents need? {Complementary} learning systems theory updated," Trends in Cognitive Sciences, vol. 20, no. 7, pp. 512--534, 2016.
```
Original content found in: `[33] Kumaran2016_ComplementaryLearningSystems.pdf` pag. 1
Citation verification details: Trends in Cognitive Sciences vol. 20, no. 7, pp. 512–534, 2016. Abstract: CLS theory "holds that intelligent agents must possess two learning systems, instantiated in mammalians in neocortex and hippocampus. The first gradually acquires structured knowledge representations while the second quickly learns specifics of individual experiences."
Metadata (Audit 2): matches Crossref (DOI) on title, authors, venue, volume, issue, pages and year.

---

[34] VERIFIED.
Paper says:
- Sec. III, p. 4: "In evolutionary computation, a hall of fame first kept the best individuals of past generations in competitive coevolution [34]."
Paper citation says:
```
[34] C. D. Rosin and R. K. Belew, "New methods for competitive coevolution," Evolutionary Computation, vol. 5, no. 1, pp. 1--29, 1997.
```
Original content found in: `[34] Rosin1997_NewMethods.pdf` pag. 2, 6 (UCSD Technical Report CS96-491, the preprint of the journal article; text not extractable, read visually)
Citation verification details: Evolutionary Computation vol. 5, no. 1, pp. 1–29, 1997. §3.3: "we introduce the hall of fame, which extends elitism in time for purposes of testing. The best individual from every generation is retained for future testing." The introduction confirms the "first" claim. The earlier conference version (Rosin & Belew, 1995) introduced only competitive fitness sharing and shared sampling, and "in this paper, we expand on these results with a new technique (the hall of fame)".
Metadata (Audit 2): matches Crossref (DOI) on title, authors, venue, volume, issue, pages and year.

---

[35] VERIFIED.
Paper says:
- Sec. III, p. 4: "Memory-enhanced evolutionary algorithms store good solutions and reinsert them when earlier conditions return [35]."
Paper citation says:
```
[35] J. Branke, "Memory enhanced evolutionary algorithms for changing optimization problems," in Proceedings of the 1999 Congress on Evolutionary Computation (CEC), vol. 3, 1999, pp. 1875--1882.
```
Original content found in: `[35] Branke1999_Memory.pdf` pag. 1–3
Citation verification details: IEEE CEC 1999, vol. 3, pp. 1875–1882. The paper surveys and extends explicit memory: storing solutions, and deciding "which individuals should be retrieved when from the memory and reinserted into the population", which pays off "when the optimum reappears at a previous location". No origin claim is made.
Metadata (Audit 2): matches Crossref (DOI) on title, authors, venue, volume, issue, pages and year (Crossref has no year; the PDF and the proceedings give 1999).

---

[36] VERIFIED.
Paper says:
- Sec. III, p. 4: "Tanh hidden layers follow the evolution-strategies policy of Salimans et al. [17], and normc initialization follows its released code [36]."
Paper citation says:
```
[36] OpenAI, "evolution-strategies-starter: Code for the paper 'Evolution strategies as a scalable alternative to reinforcement learning'," https://github.com/openai/evolution-strategies-starter, 2017.
```
Original content found in: `None (code repository, checked online)`: https://github.com/openai/evolution-strategies-starter, file `es_distributed/policies.py`
Citation verification details: The official code for [17]. The README calls it "Code for the paper 'Evolution Strategies as a Scalable Alternative to Reinforcement Learning'"; the repository has been archived (read-only) since Apr 2026. In `MujocoPolicy._make_net` the hidden layers are `self.nonlin(U.dense(x, hd, ..., U.normc_initializer(1.0)))`, which is the normc initialization the sentence credits. The continuous output is `U.dense(x, adim, 'out', U.normc_initializer(0.01))`, a linear layer, so the tanh output is credited to CEM-RL [20] instead. Checked 2026-11-14.
Metadata (Audit 2): matches the repository (README: "Code for the paper ..."; archived since Apr 2026).
---

[37] VERIFIED.
Paper says:
- Sec. III, pp. 5–6: "Fuzzy: a Mamdani controller [37] with two inputs. ... The controller's 9 rules raise Plasticity when the return is low or falling, and lower it when the return is high and steady or rising. Inference is min–max. The output is the centroid of the aggregated set, where Mamdani and Assilian took the mean of maximum. In fuzzy control, the centroid is widely used, and generally gives a lower mean square error than the mean of maximum [38]."
Paper citation says:
```
[37] E. H. Mamdani and S. Assilian, "An experiment in linguistic synthesis with a fuzzy logic controller," International Journal of Man-Machine Studies, vol. 7, no. 1, pp. 1--13, 1975.
```
Original content found in: `[37] Mamdani1975_FuzzyLogic.pdf` pag. 4–6
Citation verification details: Int. J. Man-Machine Studies vol. 7, no. 1, pp. 1–13, 1975. Rules are combined by max–min composition ("C1 = max C1′, C2′, ... = max{min ...}"), so "Mamdani controller" and "min–max" are supported. The paper makes no origin claim (Mamdani's 1974 IEE paper predates this one anyway).
Metadata (Audit 2): matches Crossref (DOI) on title, authors, venue, volume, issue, pages and year.
Fixed 2026-11-14 (was Q3, right paper): the bullet said "Inference is min–max, with centroid defuzzification", which read as part of the Mamdani controller [37]. Mamdani and Assilian took "that action... which has the largest membership grade" (the mean of maximum). The text now separates the two: [37] for the controller and min–max inference, and Lee (1990) [38] for the centroid. The academic header of `fuzzy_inference.py` was corrected the same way.

---

[38] VERIFIED.
Paper says:
- Sec. III, pp. 5–6: "Inference is min–max. The output is the centroid of the aggregated set, where Mamdani and Assilian took the mean of maximum. In fuzzy control, the centroid is widely used, and generally gives a lower mean square error than the mean of maximum [38]."
Paper citation says:
```
[38] C. C. Lee, "Fuzzy logic in control systems: Fuzzy logic controller, part II," IEEE Transactions on Systems, Man, and Cybernetics, vol. 20, no. 2, pp. 419--435, 1990.
```
Original content found in: `[38] Lee1990_FuzzyLogicControllerPartII.pdf` pag. 12 (journal p. 430)
Citation verification details: IEEE Trans. SMC vol. 20, no. 2, pp. 419–435, 1990 (doi:10.1109/21.52552). Section II, "Defuzzification Strategies", says "the commonly used strategies may be described as the max criterion, the mean of maximum, and the center of area" and "The widely used COA strategy generates the center of gravity of the possibility distribution of a control action". It reports that "An FLC based on the COA generally yields a lower mean square error than that based on the MOM", and that Braae and Rutherford "concluded that the COA strategy yields superior results". Mamdani and Assilian's rule (largest membership, midway between peaks or at the centre of a plateau) is Lee's mean of maximum. The paper's wording keeps the source's strength ("widely used", "generally").
Metadata (Audit 2): matches Crossref (DOI) on title, authors, venue, volume, issue, pages and year (Crossref writes the title as "...fuzzy logic controller. II"; the printed article says "Part II").
---

[39] VERIFIED.
Paper says:
- Sec. III, p. 7: "...where p̂t is a Fixed-Share [39] mixture of Gaussian predictors, fitted on sliding windows of 2, 4, ... recent residuals."
- Sec. III, p. 8: "BPR assumes one fixed type per trial. To follow a changing environment, a Fixed-Share step [39] mixes back a uniform share after each update: β ← 0.95 β′ + 0.05/|J|."
Paper citation says:
```
[39] M. Herbster and M. K. Warmuth, "Tracking the best expert," Machine Learning, vol. 32, no. 2, pp. 151--178, 1998.
```
Original content found in: `[39] Herbster1998_TrackingBestExpert.pdf` pag. 151–153, Fig. 1
Citation verification details: Machine Learning vol. 32, no. 2, pp. 151–178, 1998. The Fixed-share Algorithm runs a Loss Update and then a Share Update, in which "a fraction of each experts' weight is added to the weight of each other expert". Herbster and Warmuth share α of each weight among the n−1 other experts. Setting α′ = αn/(n−1) turns this into the "(1−α′)β′ + α′/n" uniform form used in the paper, so the mechanism matches. [27] also uses Fixed Share for its window weights.
Metadata (Audit 2): matches Crossref (DOI) on title, authors, venue, volume, issue, pages and year.

---

[40] VERIFIED.
Paper says:
- Sec. III, p. 7: "On HalfCheetah, ENA plays a chirp instead: a fixed probing sequence of actions, as in system identification with a known input signal [40]."
Paper citation says:
```
[40] L. Ljung, System Identification: Theory for the User, 2nd ed. Upper Saddle River, NJ: Prentice Hall, 1999.
```
Original content found in: `[40] Ljung1999_SystemIdentification.pdf` pag. 1, 409, 411, 423 (scanned with OCR; book pages, PDF pages 23, 431, 433, 445)
Citation verification details: The title page reads "System Identification: Theory for the User, Second Edition, Lennart Ljung, Linköping University, Sweden, Prentice Hall PTR, Upper Saddle River, NJ", © 1999, which matches the bib. System identification "deals with the problem of building mathematical models of dynamical systems based on observed data from the system" (p. 1). Chapter 13, "Experiment Design", treats the input as chosen by the experimenter: "The choice of input signals has a very substantial influence on the observed data. The input signals determine the operating point of the system and which parts and modes of the system are excited" (p. 409), and "the input—the 'probing signal'—should be rich. It should excite the system and force it to show its properties" (p. 411). This supports a fixed probing sequence with a known input. Section 13.3 also lists "Chirp Signals or Swept Sinusoids" as a standard input (p. 423). Ljung's chirp is a swept sinusoid, while ENA's chirp is a wavelet train played from coarse to fine scales, which also rises in frequency, in dyadic steps. The sentence cites [40] for the known input, not for the name, so this needs no change.
Metadata (Audit 3): matches the title and copyright pages (2nd edition, Prentice Hall PTR, Upper Saddle River, NJ, 1999).
---

[41] VERIFIED.
Paper says:
- Sec. III, p. 7: "Probing an environment to infer its dynamics has also been learned in RL [41]. ENA's chirp is fixed, so its response cannot depend on the agent."
Paper citation says:
```
[41] W. Zhou, L. Pinto, and A. Gupta, "Environment probing interaction policies," in International Conference on Learning Representations, 2019.
```
Original content found in: `[41] Zhou2019_EnvironmentAdaptation.pdf` pag. 1
Citation verification details: ICLR 2019 (arXiv:1907.11740). The Environment-Probing Interaction (EPI) policy is trained to "probe a new environment to extract an implicit understanding of that environment's behavior", which is then fed to a task policy.
Metadata (Audit 2): title and authors match the arXiv record; venue confirmed by the PDF header (ICLR 2019).

---

[42] VERIFIED.
Paper says:
- Sec. III, p. 7: "Each observation channel is decomposed into a Haar wavelet packet tree of depth 3 [42]."
Paper citation says:
```
[42] R. R. Coifman and M. V. Wickerhauser, "Entropy-based algorithms for best basis selection," IEEE Transactions on Information Theory, vol. 38, no. 2, pp. 713--718, 1992.
```
Original content found in: `[42] Coifman1992_Entropy.pdf` pag. 1–3 (preprint; text not extractable, read visually)
Citation verification details: IEEE Trans. Information Theory vol. 38, no. 2, pp. 713–718, 1992. The paper defines "the wavelet packet library" by recursive splitting with the filters F0/F1. Each library "has a natural dyadic tree structure". ENA uses the full tree, not the best-basis search that is the paper's main topic. The tree itself is defined here, so the citation fits. No origin claim is made.
Metadata (Audit 2): matches Crossref (DOI) on title, authors, venue, volume, issue, pages and year.

---

[43] VERIFIED.
Paper says:
- Sec. III, p. 7: "Like the discrete wavelet transform [43], the tree is dyadic and non-overlapping, but it splits every band."
Paper citation says:
```
[43] S. G. Mallat, "A theory for multiresolution signal decomposition: The wavelet representation," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 11, no. 7, pp. 674--693, 1989.
```
Original content found in: `[43] Mallat1989_WaveletTheory.pdf` pag. 674–679
Citation verification details: IEEE TPAMI vol. 11, no. 7, pp. 674–693, 1989. Multiresolution decomposition: the dyadic discrete wavelet transform recursively splits only the approximation (low-pass) band.
Metadata (Audit 2): matches Crossref (DOI) on title, authors, venue, volume, issue, pages and year.

---

[44] VERIFIED.
Paper says:
- Sec. III, p. 7: "...one log-energy is kept per band. In machine fault diagnosis, the energies of wavelet-packet bands are widely used to characterize faults [44]."
Paper citation says:
```
[44] R. Yan, R. X. Gao, and X. Chen, "Wavelets for fault diagnosis of rotary machines: {A} review with applications," Signal Processing, vol. 96, pp. 1--15, 2014.
```
Original content found in: `[44] Yan2014_WaveletsFaultDiagnosis.pdf` pag. 1, 8, 10
Citation verification details: Signal Processing vol. 96, pp. 1–15, 2014 (doi:10.1016/j.sigpro.2013.04.015). A review of wavelets for rotary-machine fault diagnosis (bearings, gearboxes, rotors). It supports wavelet features for machine faults, including band energies like ENA's: "features extracted from wavelet coefficients of the WPT have also been widely used for characterizing machine faults. For example, Zarei and Poshtan applied the WPT to induction motor stator current and extracted sub-frequency band energy as fault index".
Metadata (Audit 2): matches Crossref (DOI) on title, authors, venue, volume, issue, pages and year.
Fixed 2026-11-14 (was Q1 and Q3): the sentence said "Wavelet features of a system's response to a known excitation are a standard signature of a change of physics, in structural damage detection and machine fault diagnosis". The source says "widely used", never mentions an excitation, and covers rotary machines only. The sentence now says only what the source reports: wavelet-packet band energies are widely used to characterize machine faults. The structural-engineering claim was dropped. Its unused bib entry (`kijewski2003wavelet`) was deleted on 2026-11-15.

---

[45] VERIFIED.
Paper says:
- Sec. III, p. 7: "The transforms are computed with PyWavelets [45]."
Paper citation says:
```
[45] G. R. Lee, R. Gommers, F. Waselewski, K. Wohlfahrt, and A. O'Leary, "{PyWavelets}: A {Python} package for wavelet analysis," Journal of Open Source Software, vol. 4, no. 36, p. 1237, 2019.
```
Original content found in: `[45] Lee2019_PyWavelets.pdf` pag. 1
Citation verification details: JOSS vol. 4, no. 36, p. 1237, 2019. This is the software citation for the library that computes the transforms.
Metadata (Audit 2): matches Crossref (DOI) on title, authors, venue, volume, issue, pages and year.

---

[46] VERIFIED.
Paper says:
- Sec. IV, p. 8: "ENA was evaluated in two non-stationary domains built on Gymnasium [46]: the discrete-action CartPole-v1 [11], and the continuous-control MuJoCo [12] HalfCheetah-v5."
Paper citation says:
```
[46] M. Towers, A. Kwiatkowski, J. Terry, J. U. Balis, G. De Cola, T. Deleu, M. Goul{\ a}o, A. Kallinteris, M. Krimmel, A. KG, R. Perez-Vicente, A. Pierr{\'e}, S. Schulhoff, J. J. Tai, H. Tan, and O. G. Younis, "Gymnasium: A standard interface for reinforcement learning environments," arXiv preprint arXiv:2407.17032, 2024.
```
Original content found in: `[46] Towers2024_Gymnasium.pdf` pag. 1
Citation verification details: arXiv:2407.17032. Gymnasium is the maintained successor of OpenAI Gym and provides both environments.
Metadata (Audit 2): title and authors match the arXiv record. The rendered entry is correct (M. Goulão, A. Pierré).

---

[47] VERIFIED.
Paper says:
- Sec. IV, p. 9: "On CartPole, ENA-Combined was compared with DQN [13], PPO [14] and the discrete-action variant of SAC [47]."
Paper citation says:
```
[47] P. Christodoulou, "Soft actor-critic for discrete action settings," arXiv preprint arXiv:1910.07207, 2019.
```
Original content found in: `[47] Christodoulou2019_SACDiscrete.pdf` pag. 2–3
Citation verification details: arXiv:1910.07207. Section 3 derives SAC for discrete actions (SAC-Discrete).
Metadata (Audit 2): title and authors match the arXiv record.

---

[48] VERIFIED.
Paper says:
- Sec. V, p. 14: "The Haar and db4 [48] trains depend strongly on the features."
Paper citation says:
```
[48] I. Daubechies, "Orthonormal bases of compactly supported wavelets," Communications on Pure and Applied Mathematics, vol. 41, no. 7, pp. 909--996, 1988.
```
Original content found in: `[48] Daubechies1988_OrthonormalBases.pdf` pag. 909, Table 1
Citation verification details: Comm. Pure Appl. Math. vol. 41, no. 7, pp. 909–996, 1988. Table 1 lists the filter coefficients for N = 2, ..., 10. N = 4 is the db4 wavelet (8 taps) used by PyWavelets.
Metadata (Audit 2): matches Crossref (DOI) on title, authors, venue, volume, issue, pages and year.

---

[49] VERIFIED.
Paper says:
- Sec. VI, p. 17: "The cost of exploring in parameter space grows with the number of parameters [49], and random mutations tend to disrupt large networks [50]."
- Sec. VII, p. 18: "Both limitations align with known properties of evolutionary search, which exhibits high sample complexity and scales unfavorably with parameter count [19], [49]."
Paper citation says:
```
[49] A. Vemula, W. Sun, and J. A. Bagnell, "Contrasting exploration in parameter and action space: A zeroth-order optimization perspective," in Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics (AISTATS), ser. Proceedings of Machine Learning Research, vol. 89, 2019, pp. 2926--2935.
```
Original content found in: `[49] Vemula2019_Contrasting.pdf` pag. 1, 4–6
Citation verification details: AISTATS 2019 (PMLR vol. 89, pp. 2926–2935; arXiv:1901.11503). Abstract: "we prove that complexity of exploration in parameter space depends on the dimensionality of parameter space".
Metadata (Audit 2): title and authors match the arXiv record; the PMLR page confirms PMLR 89:2926–2935.

---

[50] VERIFIED.
Paper says:
- Sec. VI, p. 17: "The cost of exploring in parameter space grows with the number of parameters [49], and random mutations tend to disrupt large networks [50]. This is why ENA's specialists are small."
Paper citation says:
```
[50] J. Lehman, J. Chen, J. Clune, and K. O. Stanley, "Safe mutations for deep and recurrent neural networks through output gradients," in Proceedings of the Genetic and Evolutionary Computation Conference (GECCO), 2018.
```
Original content found in: `[50] Lehman2018_SafeMutations.pdf` pag. 1
Citation verification details: GECCO 2018 (arXiv:1712.06563). Abstract: "while random mutation generally works in low dimensions, a random perturbation of thousands or millions of weights will likely break existing functionality."
Metadata (Audit 2): matches Crossref (DOI) on title, authors, venue, volume, issue, pages and year; title and authors match the arXiv record.

---

[51] VERIFIED.
Paper says:
- Appendix, p. 22: "On HalfCheetah, PPO and SAC take the settings of RL Baselines3 Zoo [51] (Table XXIII). Both have a separate actor and critic. PPO normalizes observations, and scales rewards by the standard deviation of the return. PPO keeps the zoo's constant learning rate. One setting deviates from the zoo: SAC applies its gradient steps in groups of 20 every 20 environment steps, for speed, which preserves the zoo's ratio of one update per step."
Paper citation says:
```
[51] A. Raffin, "{RL} baselines3 zoo," https://github.com/DLR-RM/rl-baselines3-zoo, 2020.
```
Original content found in: `[51] Raffin2020_RLZoo.pdf` pag. 2
Citation verification details: The cited work is the GitHub repository. The local PDF is a different paper, the Stable-Baselines3 JMLR article (vol. 22, no. 268, 2021). It cites "Antonin Raffin. RL Baselines3 Zoo. https://github.com/DLR-RM/rl-baselines3-zoo, 2020" and says the Zoo provides "agents together with tuned hyperparameters". The citation is right. Audit 2 compared the values with the Zoo's `hyperparams/ppo.yml` and `sac.yml` (HalfCheetah-v4 entries, fetched 2026-11-14). PPO matches every listed setting: n_steps 512, batch 64, 20 epochs, γ 0.98, λ 0.92, clip 0.1, lr 2.0633e-5, entropy 0.000401762, value 0.58096, grad norm 0.8, log std −2, no orthogonal init, separate 256×256 ReLU networks, `normalize: true`. SAC uses the Stable-Baselines3 defaults with `learning_starts: 10000`, as the Zoo does. The SB3 article is not cited.
Metadata (Audit 2): the repository URL and year match the Stable-Baselines3 JMLR paper, which cites the same work.

---
Fixed 2026-11-14 (Audit 2, was Q4, description): the appendix listed "PPO's learning rate is not annealed" as a deviation from the Zoo. The Zoo's tuned HalfCheetah PPO already uses a constant `learning_rate: 2.0633e-05`, so it is the Zoo's own setting. The appendix now says "PPO keeps the zoo's constant learning rate. One setting deviates from the zoo: SAC applies its gradient steps in groups...".

---

## Claims without a citation (Audit 2)

These sentences make a claim about the literature or about common practice, but cite nothing. Item 1 was fixed. Items 2–6 are optional. They are not numbered citations, so they have no status above. Fixes are in [citation_fix_plan.md](citation_fix_plan.md), "Audit 2".

1. **Fixed 2026-11-14. Sec. IV, p. 9 (CartPole baselines):** the sentence now reads "All are custom implementations, with the default hyperparameters of these implementations (Appendix A)." It used to say "with common hyperparameters for these benchmarks". Nothing supports "common". The code says the baselines use "their implementations' default hyperparameters". CartPole PPO uses the general Stable-Baselines3 defaults (lr 3e-4, 2048 steps, 10 epochs, λ 0.95, clip 0.2, 64×64 tanh) plus entropy 0.01, not tuned CartPole values. The DQN settings match neither the Zoo's tuned CartPole entry nor the Stable-Baselines3 defaults.
2. **Sec. IV, p. 9:** "With the usual linear annealing to zero over the training budget..." This is supported but uncited. The Zoo's tuned CartPole-v1 PPO uses `learning_rate: lin_0.001`, and the PPO paper [14] anneals linearly on Atari (Table 5: "α is linearly annealed from 1 to 0"). On MuJoCo, [14] uses a constant step size, so "usual" holds for CartPole but is not universal.
3. **Sec. IV, p. 9 (SAC-Discrete):** "...instead of 0.98 log|A|". The 0.98 is [47]'s value (Table 2: "Entropy target 0.98 * (-log (1/|A|))"). [47] is cited two sentences earlier, so this is fine. Citing it again here is optional.
4. **Sec. IV and V (statistics):** Holm's method, the Wilcoxon signed-rank test, Welch's t-test, the exact McNemar test, the paired effect size d_z and the power analysis are named without citations. These are standard, and many venues do not require citations for them. Holm (1979) is the one most often cited.
5. **Sec. III, p. 4:** "orthogonal initialization" (CartPole specialists) is uncited. This is optional (Saxe et al., 2014, is the usual reference).
6. **Sec. I, p. 1:** "Reinforcement learning (RL) agents are usually trained and deployed on the assumption that the environment does not change." This is a general framing sentence. [2] could support it, which is optional.
