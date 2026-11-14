# Citations Verification

This document verifies each of the 49 citations in the paper against the manuscript content, the compiled bibliography entry, and the reference documents in `references/`.

---

[1] UNVERIFIED!!!
Paper says: "When an agent with a single policy adapts by overwriting the parameters that encoded its previous behavior. This is catastrophic forgetting [1]." Pag. 1
Paper citation says:
```
[1] M. McCloskey and N. J. Cohen, "Catastrophic interference in connectionist networks: The sequential learning problem," in Psychology of Learning and Motivation. Academic Press, 1989, vol. 24, pp. 109--165.
```
Original content found in: `None (Missing from references folder)`
Citation verification details: The PDF is unavailable in references/ because the publication is a copyrighted book chapter in 'Psychology of Learning and Motivation', Vol. 24, Academic Press (1989), pp. 109–165. Metadata is verified via CrossRef DOI 10.1016/S0079-7421(08)60536-8.

---

[2] VERIFIED.
Paper says: "Continual reinforcement learning studies agents that keep adapting without losing what they have already learned [2]." Pag. 1
Paper citation says:
```
[2] K. Khetarpal, M. Riemer, I. Rish, and D. Precup, "Towards continual reinforcement learning: A review and perspectives," Journal of Artificial Intelligence Research, vol. 75, pp. 1401--1476, 2022.
```
Original content found in: `[02] Khetarpal2022_ContinualRL.pdf` pag. 1
Citation verification details: The header of the PDF states 'Journal of Artificial Intelligence Research, vol. 75, pp. 1401–1476, 2022'. Section 1 ('Introduction') on page 1 defines continual reinforcement learning as agents acquiring and transferring knowledge across multiple non-stationary tasks without catastrophic forgetting.

---

[3] VERIFIED.
Paper says: "One family of methods protects or partitions the weights of a shared network, by regularization [3]..." Pag. 1
Paper citation says:
```
[3] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska, D. Hassabis, C. Clopath, D. Kumaran, and R. Hadsell, "Overcoming catastrophic forgetting in neural networks," Proceedings of the National Academy of Sciences, vol. 114, no. 13, pp. 3521--3526, 2017.
```
Original content found in: `[03] Kirkpatrick2017_OvercomingCatastrophicForgetting.pdf` pag. 2
Citation verification details: The paper introduces Elastic Weight Consolidation (EWC) on page 2 ('protecting weights that are important for previous tasks by adding a quadratic penalty') and matches PNAS vol. 114, no. 13, pp. 3521–3526, 2017.

---

[4] VERIFIED.
Paper says: "...replay [4] or tasks with dedicated modules..." Pag. 1
Paper citation says:
```
[4] D. Rolnick, A. Ahuja, J. Schwarz, T. Lillicrap, and G. Wayne, "Experience replay for continual learning," in Advances in Neural Information Processing Systems, vol. 32, 2019.
```
Original content found in: `[04] Rolnick2019_ExperienceReplay.pdf` pag. 2
Citation verification details: Section 2 on pages 2–3 introduces CLEAR (Continual Learning with Experience And Replay), using experience replay to prevent forgetting without needing task boundaries; matches NeurIPS 2019, vol. 32.

---

[5] VERIFIED.
Paper says: "For such tasks, OWL [5] gives each task its own policy head on a shared feature extractor, and chooses a head at test time with a multi-armed bandit." Pag. 1
Paper citation says:
```
[5] S. Kessler, J. Parker-Holder, P. Ball, S. Zohren, and S. J. Roberts, "Same state, different task: Continual reinforcement learning without interference," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36, no. 7, 2022, pp. 7143--7151.
```
Original content found in: `[05] Kessler2022_SameEnvironment.pdf` pag. 1
Citation verification details: Section 1 and 3 on pages 1–3 introduce OWL, assigning dedicated policy heads on a shared base and using a bandit at test time; matches AAAI 2022, vol. 36, no. 7, pp. 7143–7151.

---

[6] VERIFIED.
Paper says: "ENA, like the context-based methods below, preserves whole policies instead... Context-based methods identify which task the environment is currently presenting [6]..." Pag. 1
Paper citation says:
```
[6] B. C. da Silva, E. W. Basso, A. L. C. Bazzan, and P. M. Engel, "Dealing with non-stationary environments using context detection," in Proceedings of the 23rd International Conference on Machine Learning, 2006, pp. 217--224.
```
Original content found in: `[06] daSilva2006_NonStationaryRL.pdf` pag. 1
Citation verification details: Section 1 and 3 on pages 1–3 present context detection in non-stationary RL by monitoring environment dynamics to switch or maintain policies; matches ICML 2006, pp. 217–224.

---

[7] VERIFIED.
Paper says: "MBCD [7] combined an ensemble of learned models with change-point detection on their prediction errors to detect switches in transition dynamics..." Pag. 3
Paper citation says:
```
[7] L. N. Alegre, A. L. C. Bazzan, and B. C. da Silva, "Minimum-delay adaptation in non-stationary reinforcement learning via online high-confidence change-point detection," in Proceedings of the 20th International Conference on Autonomous Agents and Multiagent Systems (AAMAS), 2021, pp. 97--105.
```
Original content found in: `[07] Alegre2021_MinimumWidth.pdf` pag. 2
Citation verification details: Section 3 and 4 on pages 2–4 present Model-Based Change Detection (MBCD) using high-confidence change-point detection on transition errors; matches AAMAS 2021, pp. 97–105.

---

[8] UNVERIFIED!!!
Paper says: "Other context-detection approaches evaluate predictions from an ensemble of world models or statistical tests over observations [8]..." Pag. 1
Paper citation says:
```
[8] J. Dick, S. Nath, C. Peridis, E. Benjamin, S. Kolouri, and A. Soltoggio, "Statistical context detection for deep lifelong reinforcement learning," in Proceedings of the 3rd Conference on Lifelong Learning Agents (CoLLAs), 2024.
```
Original content found in: `[08] Dick2024_Statistical.pdf`
Citation verification details: The file downloaded as '[08] Dick2024_Statistical.pdf' contains an unrelated paper titled 'TADA: Temporal Adversarial Data Augmentation for Time Series Data' by Byeong Tak Lee et al. (arXiv:2407.15174). The true paper is 'Statistical context detection for deep lifelong reinforcement learning' by Jeffery Dick et al. in CoLLAs 2024 (PMLR v274, pp. 1013–1031).

---

[9] VERIFIED.
Paper says: "Nisioti et al. [9] showed that neuroevolution can preserve behaviors in continual RL when combined with quality-diversity..." Pag. 1
Paper citation says:
```
[9] E. Nisioti, A. Cossu, K. Korte, and S. Risi, "Continual reinforcement learning with neuroevolution," arXiv preprint arXiv:2610.01583, 2026.
```
Original content found in: `[09] Nisioti2026_ContinualQD.pdf` pag. 1
Citation verification details: Section 1 and 2 on pages 1–4 study continual reinforcement learning with neuroevolution and show how population-based QD preserves behaviors across changing environments; matches arXiv:2610.01583.

---

[10] VERIFIED.
Paper says: "...governs which archived specialist to deploy via Bayesian policy reuse [10], integrating detector evidence with observed episodic returns." Pag. 2
Paper citation says:
```
[10] B. Rosman, M. Hawasly, and S. Ramamoorthy, "Bayesian policy reuse," Machine Learning, vol. 104, no. 1, pp. 99--127, 2016.
```
Original content found in: `[10] Rosman2016_BayesianPolicyReuse.pdf` pag. 3
Citation verification details: Section 3 on pages 3–5 formulates Bayesian Policy Reuse (BPR) to select from an archive of previously learned policies based on observed performance; matches Machine Learning (Springer), vol. 104, 2016.

---

[11] VERIFIED.
Paper says: "Two domains are studied. In CartPole [11], the agent balances a pole on a moving cart." Pag. 2
Paper citation says:
```
[11] A. G. Barto, R. S. Sutton, and C. W. Anderson, "Neuronlike adaptive elements that can solve difficult learning control problems," IEEE Transactions on Systems, Man, and Cybernetics, vol. SMC-13, no. 5, pp. 834--846, 1983.
```
Original content found in: `[11] Barto1983_NeuronLike.pdf` pag. 1
Citation verification details: The cover sheet and Section 2 confirm Barto, Sutton & Anderson's classic cart-pole / inverted pendulum balancing problem in IEEE Trans. Systems, Man, and Cybernetics, vol. SMC-13, pp. 834–846, 1983.

---

[12] VERIFIED.
Paper says: "In HalfCheetah [12], a simulated planar robot learns to run under varied gravity, torso mass, and joint friction." Pag. 2
Paper citation says:
```
[12] E. Todorov, T. Erez, and Y. Tassa, "{MuJoCo}: A physics engine for model-based control," in 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, 2012, pp. 5026--5033.
```
Original content found in: `[12] Todorov2012_MuJoCo.pdf` pag. 1
Citation verification details: Page 1 introduces MuJoCo as a physics engine for model-based control and multi-joint locomotion benchmark environments; matches IEEE IROS 2012, pp. 5026–5033.

---

[13] UNVERIFIED!!!
Paper says: "...with the same number of environment steps: DQN [13], PPO [14] and SAC [15]." Pag. 2
Paper citation says:
```
[13] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis, "Human-level control through deep reinforcement learning," Nature, vol. 518, no. 7540, pp. 529--533, 2015.
```
Original content found in: `[13] Mnih2015_HumanLevelDQN.pdf`
Citation verification details: The file stored in references/ is the earlier 2013 NIPS workshop preprint ('Playing Atari with Deep Reinforcement Learning', arXiv:1312.5602), whereas the citation is the landmark 2015 Nature article ('Human-level control through deep reinforcement learning', Nature, vol. 518, no. 7540, pp. 529–533, 2015).

---

[14] VERIFIED.
Paper says: "...DQN [13], PPO [14] and SAC [15]." Pag. 2
Paper citation says:
```
[14] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," arXiv preprint arXiv:1707.06347, 2017.
```
Original content found in: `[14] Schulman2017_PPO.pdf` pag. 2
Citation verification details: Section 3 on page 3 introduces Proximal Policy Optimization (PPO) with the clipped surrogate objective; matches Schulman et al., arXiv:1707.06347, 2017.

---

[15] VERIFIED.
Paper says: "...DQN [13], PPO [14] and SAC [15]." Pag. 2
Paper citation says:
```
[15] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor," in Proceedings of the 35th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 80, 2018, pp. 1861--1870.
```
Original content found in: `[15] Haarnoja2018_SAC.pdf` pag. 2
Citation verification details: Section 3 and 4 on pages 2–5 formulate Soft Actor-Critic (SAC) maximizing expected return plus entropy; matches ICML 2018 (PMLR v80), pp. 1861–1870.

---

[16] UNVERIFIED!!!
Paper says: "Neuroevolution optimizes neural networks by mutation and selection over a population, instead of by gradients [16]." Pag. 2
Paper citation says:
```
[16] K. O. Stanley, J. Clune, J. Lehman, and R. Miikkulainen, "Designing neural networks through neuroevolution," Nature Machine Intelligence, vol. 1, no. 1, pp. 24--35, 2019.
```
Original content found in: `[16] Stanley2019_DesigningNeuralNetworks.pdf`
Citation verification details: The file currently in references/ is an unrelated physics paper ('Optical vs electronic gap of hafnia' by Sklénard et al., arXiv:1811.02362). The true paper is 'Designing neural networks through neuroevolution' by Stanley, Clune, Lehman & Miikkulainen in Nature Machine Intelligence, vol. 1, pp. 24–35, 2019.

---

[17] VERIFIED.
Paper says: "Evolution strategies [17] and genetic algorithms [18] are competitive with policy gradients on standard reinforcement learning benchmarks..." Pag. 2
Paper citation says:
```
[17] T. Salimans, J. Ho, X. Chen, S. Sidor, and I. Sutskever, "Evolution strategies as a scalable alternative to reinforcement learning," arXiv preprint arXiv:1703.03864, 2017.
```
Original content found in: `[17] Salimans2017_EvolutionStrategies.pdf` pag. 1
Citation verification details: Page 1 ('Abstract' and 'Introduction') demonstrates Evolution Strategies (ES) as a competitive, highly scalable alternative to policy gradient reinforcement learning on MuJoCo benchmarks; matches arXiv:1703.03864, 2017.

---

[18] VERIFIED.
Paper says: "...and genetic algorithms [18] are competitive with policy gradients on standard reinforcement learning benchmarks..." Pag. 2
Paper citation says:
```
[18] F. P. Such, V. Madhavan, E. Conti, J. Lehman, K. O. Stanley, and J. Clune, "Deep neuroevolution: Genetic algorithms are a competitive alternative for training deep neural networks for reinforcement learning," arXiv preprint arXiv:1712.06567, 2017.
```
Original content found in: `[18] Such2017_DeepNeuroevolution.pdf` pag. 1
Citation verification details: Page 1 demonstrates that a basic genetic algorithm (GA) can train deep neural networks for Atari and continuous control benchmarks; matches Such et al., arXiv:1712.06567, 2017.

---

[19] VERIFIED.
Paper says: "They need many samples, however, and optimizing many parameters remains hard for them [19], [20]." Pag. 2
Paper citation says:
```
[19] S. Khadka and K. Tumer, "Evolution-guided policy gradient in reinforcement learning," in Advances in Neural Information Processing Systems, vol. 31, 2018, pp. 1188--1200.
```
Original content found in: `[19] Khadka2018_EvolutionGuidedPolicyGradient.pdf` pag. 1
Citation verification details: Section 1 on pages 1–2 discusses the sample inefficiency and parameter scaling challenges of neuroevolution and introduces Evolutionary Reinforcement Learning (ERL); matches NeurIPS 2018, vol. 31.

---

[20] VERIFIED.
Paper says: "They need many samples, however, and optimizing many parameters remains hard for them [19], [20]." Pag. 2
Paper citation says:
```
[20] A. Pourchot and O. Sigaud, "{CEM-RL}: Combining evolutionary and gradient-based methods for policy search," in International Conference on Learning Representations (ICLR), 2019.
```
Original content found in: `[20] Pourchot2019_CEMRL.pdf` pag. 1
Citation verification details: Section 1 on page 1 analyzes parameter search challenges in high dimensions and introduces CEM-RL combining cross-entropy methods with DDPG/TD3; matches ICLR 2019 (arXiv:1810.01222).

---

[21] VERIFIED.
Paper says: "Earlier work on evolutionary algorithms for changing optimization problems developed remedies such as increasing diversity after a change, maintaining diversity throughout the run, and memory [21]." Pag. 2
Paper citation says:
```
[21] D. Yazdani, R. Cheng, D. Yazdani, J. Branke, Y. Jin, and X. Yao, "A survey of evolutionary continuous dynamic optimization over two decades---part {A}," IEEE Transactions on Evolutionary Computation, vol. 25, no. 4, pp. 609--629, 2021.
```
Original content found in: `[21] Yazdani2021_SurveyDynamicOptimization.pdf` pag. 1
Citation verification details: Page 1 and Section II/III review two decades of continuous dynamic optimization in evolutionary computation, categorizing remedies into diversity maintenance, diversity increase after change, and memory mechanisms; matches IEEE TEVC, vol. 25, no. 4, pp. 609–629, 2021.

---

[22] VERIFIED.
Paper says: "Quality-Diversity algorithms such as MAP-Elites [22] search for many diverse, high-performing behaviors..." Pag. 2
Paper citation says:
```
[22] J.-B. Mouret and J. Clune, "Illuminating search spaces by mapping elites," arXiv preprint arXiv:1504.04909, 2015.
```
Original content found in: `[22] Mouret2015_IlluminatingSearch.pdf` pag. 1
Citation verification details: Pages 1–3 introduce the Multi-dimensional Archive of Phenotypic Elites (MAP-Elites) algorithm to find diverse, high-performing behavioral niches; matches Mouret & Clune, arXiv:1504.04909, 2015.

---

[23] VERIFIED.
Paper says: "Intelligent Trial and Error (ITE) [23] computes such an archive offline, and uses it to adapt to physical damage in a physical robot..." Pag. 2
Paper citation says:
```
[23] A. Cully, J. Clune, D. Tarapore, and J.-B. Mouret, "Robots that can adapt like animals," Nature, vol. 521, no. 7553, pp. 503--507, 2015.
```
Original content found in: `[23] Cully2015_RobotsAdapt.pdf` pag. 1
Citation verification details: Page 1 (Nature manuscript) introduces the Intelligent Trial and Error (IT&E) algorithm enabling damaged robots to recover locomotion in under two minutes by searching an offline MAP-Elites repertoire; matches Nature, vol. 521, no. 7553, pp. 503–507, 2015.

---

[24] VERIFIED.
Paper says: "TeLAPA [24] brings the same idea to continual RL in discrete tasks, using an archive of policies to preserve plasticity..." Pag. 2
Paper citation says:
```
[24] L. Lillo and N. Cheney, "Beyond single-model optimization: Preserving plasticity in continual reinforcement learning," in Proceedings of the 5th Conference on Lifelong Learning Agents (CoLLAs), 2026, arXiv:2604.15414.
```
Original content found in: `[24] Lillo2026_BeyondSingleTask.pdf` pag. 2
Citation verification details: Section 1 and 3 on pages 2–4 introduce Temporal Layer-Adaptive Plasticity Architecture (TeLAPA) preserving plasticity across continual learning tasks; matches CoLLAs 2026.

---

[25] VERIFIED.
Paper says: "Classic methods range from Page’s CUSUM [25] to Bayesian online changepoint detection [26]." Pag. 3
Paper citation says:
```
[25] E. S. Page, "Continuous inspection schemes," Biometrika, vol. 41, no. 1--2, pp. 100--115, 1954.
```
Original content found in: `[25] Page1954_ContinuousSampling.pdf` pag. 1
Citation verification details: Page 100 introduces continuous inspection schemes and the cumulative sum (CUSUM) decision function for detecting shifts in distributions; matches Biometrika, vol. 41, no. 1/2, pp. 100–115, 1954.

---

[26] VERIFIED.
Paper says: "...to Bayesian online changepoint detection [26]." Pag. 3
Paper citation says:
```
[26] R. P. Adams and D. J. C. MacKay, "Bayesian online changepoint detection," arXiv preprint arXiv:0710.3742, 2007.
```
Original content found in: `[26] Adams2007_BayesianChangepoint.pdf` pag. 1
Citation verification details: Section 1 and 2 on pages 1–2 formulate exact Bayesian Online Changepoint Detection (BOCPD) using message-passing on run-length distributions; matches Adams & MacKay, arXiv:0710.3742, 2007.

---

[27] VERIFIED.
Paper says: "On CartPole, ENA uses the Predictive-Mixture CuSum [27]. It detects a change when the likelihood of a test statistic under the predictive distribution drops..." Pag. 3
Paper citation says:
```
[27] T. Halme, H. V. Poor, and V. Koivunen, "Adaptive sequential change detection using mixtures of predictive distributions," arXiv preprint arXiv:2606.05072, 2026.
```
Original content found in: `[27] Halme2026_SequentialChangepoint.pdf` pag. 2
Citation verification details: Section 2 and 3 on pages 2–5 develop sequential change detection using mixtures of predictive distributions (PM-CuSum); matches Halme, Poor & Koivunen, arXiv:2606.05072, 2026.

---

[28] UNVERIFIED!!!
Paper says: "Variation rises when the current genotype is failing and stays low otherwise, because most mutations harm a well-adapted organism [28]." Pag. 4
Paper citation says:
```
[28] R. S. Galhardo, P. J. Hastings, and S. M. Rosenberg, "Mutation as a stress response and the regulation of evolvability," Critical Reviews in Biochemistry and Molecular Biology, vol. 42, no. 5, pp. 399--435, 2007.
```
Original content found in: `[28] Galhardo2007_MutationRates.pdf`
Citation verification details: The file currently in references/ is an unrelated PLoS Genetics paper ('Unexpected Novel Relational Links Uncovered by Extensive Developmental Profiling...'). The true cited paper is 'Mutation as a stress response and the regulation of evolvability' by R. S. Galhardo, P. J. Hastings & S. M. Rosenberg in Critical Reviews in Biochemistry and Molecular Biology, vol. 42, no. 5, pp. 399–435, 2007.

---

[29] VERIFIED.
Paper says: "Evolutionary computation adopted the idea as triggered hypermutation, which raises the mutation rate when performance drops in a non-stationary environment [29]." Pag. 4
Paper citation says:
```
[29] H. G. Cobb, "An investigation into the use of hypermutation as an adaptive operator in genetic algorithms having continuous, time-dependent nonstationary environments," Naval Research Laboratory, Washington, DC, Tech. Rep. AIC-90-001, 1990.
```
Original content found in: `[29] Cobb1990_Investigation.pdf` pag. 4
Citation verification details: Section 2 on pages 4–6 introduces the 'Triggered Hypermutation' operator that monitors time-averaged performance in non-stationary environments and elevates mutation rates upon performance decline; matches NRL Memorandum Report 6760, 1990.

---

[30] UNVERIFIED!!!
Paper says: "More generally, it adopted the idea of controlling an algorithm’s parameters during the run [30]." Pag. 4
Paper citation says:
```
[30] G. Karafotias, M. Hoogendoorn, and A. E. Eiben, "Parameter control in evolutionary algorithms: Trends and challenges," IEEE Transactions on Evolutionary Computation, vol. 19, no. 2, pp. 167--187, 2015.
```
Original content found in: `None (Missing from references folder)`
Citation verification details: The PDF is unavailable in references/ because the publication is behind IEEE paywall ('Parameter Control in Evolutionary Algorithms: Trends and Challenges', IEEE Transactions on Evolutionary Computation, vol. 19, no. 2, pp. 167–187, 2015, DOI: 10.1109/TEVC.2014.2308294).

---

[31] UNVERIFIED!!!
Paper says: "Freshly initialized specialists also enter the population in every generation, as random immigrants [31] do." Pag. 4
Paper citation says:
```
[31] J. J. Grefenstette, "Genetic algorithms for changing environments," in Parallel Problem Solving from Nature 2, R. M{\"a}nner and B. Manderick, Eds. North-Holland, 1992, pp. 137--144.
```
Original content found in: `None (Missing from references folder)`
Citation verification details: The PDF is unavailable in references/ because the publication is a paywalled book chapter in 'Parallel Problem Solving from Nature 2' (Elsevier, 1992, pp. 483–492, 'Genetic algorithms for changing environments').

---

[32] VERIFIED.
Paper says: "The complementary learning systems theory [32], [33] holds that the brain learns quickly in the hippocampus and consolidates slowly into the neocortex." Pag. 4
Paper citation says:
```
[32] J. L. McClelland, B. L. McNaughton, and R. C. O'Reilly, "Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and memory," Psychological Review, vol. 102, no. 3, pp. 419--457, 1995.
```
Original content found in: `[32] McClelland1995_ComplementaryLearning.pdf` pag. 1
Citation verification details: Section 1 and 2 on pages 419–422 state the complementary learning systems (CLS) theory: rapid hippocampal learning protects previously acquired neocortical knowledge from catastrophic interference; matches Psychological Review, vol. 102, no. 3, pp. 419–457, 1995.

---

[33] VERIFIED.
Paper says: "The complementary learning systems theory [32], [33] holds that the brain learns quickly in the hippocampus and consolidates slowly into the neocortex." Pag. 4
Paper citation says:
```
[33] D. Kumaran, D. Hassabis, and J. L. McClelland, "What learning systems do intelligent agents need? {Complementary} learning systems theory updated," Trends in Cognitive Sciences, vol. 20, no. 7, pp. 512--534, 2016.
```
Original content found in: `[33] Kumaran2016_ComplementaryLearningSystems.pdf` pag. 3
Citation verification details: Pages 3–8 update the CLS framework and link hippocampal replay directly to experience replay in deep reinforcement learning (DQN); matches Trends in Cognitive Sciences, vol. 20, no. 7, pp. 512–534, 2016.

---

[34] VERIFIED.
Paper says: "In evolutionary computation, a hall of fame first kept the best individuals of past generations in competitive coevolution [34]." Pag. 4
Paper citation says:
```
[34] C. D. Rosin and R. K. Belew, "New methods for competitive coevolution," Evolutionary Computation, vol. 5, no. 1, pp. 1--29, 1997.
```
Original content found in: `[34] Rosin1997_NewMethods.pdf` pag. 2
Citation verification details: Section 2 on pages 2–4 introduces the 'Hall of Fame' technique to prevent cyclical dynamics and preserve best individuals across generations in competitive coevolution; matches Evolutionary Computation (MIT Press), vol. 5, no. 1, pp. 1–29, 1997.

---

[35] VERIFIED.
Paper says: "Memory-enhanced evolutionary algorithms store good solutions and reinsert them when earlier conditions return [35]." Pag. 4
Paper citation says:
```
[35] J. Branke, "Memory enhanced evolutionary algorithms for changing optimization problems," in Proceedings of the 1999 Congress on Evolutionary Computation (CEC), vol. 3, 1999, pp. 1875--1882.
```
Original content found in: `[35] Branke1999_Memory.pdf` pag. 1
Citation verification details: Section 1 and 2 on pages 1–2 define memory-enhanced evolutionary algorithms storing solutions in an explicit memory buffer and re-evaluating/reinserting them when environmental changes recur; matches IEEE CEC 1999, vol. 3, pp. 1875–1882.

---

[36] VERIFIED.
Paper says: "• Fuzzy: a Mamdani controller [36] with two inputs." Pag. 5
Paper citation says:
```
[36] E. H. Mamdani and S. Assilian, "An experiment in linguistic synthesis with a fuzzy logic controller," International Journal of Man-Machine Studies, vol. 7, no. 1, pp. 1--13, 1975.
```
Original content found in: `[36] Mamdani1975_FuzzyLogic.pdf` pag. 1
Citation verification details: Pages 1–4 detail the linguistic synthesis of a fuzzy logic controller using max-min composition rules; matches Int. J. Man-Machine Studies, vol. 7, no. 1, pp. 1–13, 1975.

---

[37] VERIFIED.
Paper says: "...where p̂t is a Fixed-Share [37] mixture of Gaussian predictors..." Pag. 6
Paper citation says:
```
[37] M. Herbster and M. K. Warmuth, "Tracking the best expert," Machine Learning, vol. 32, no. 2, pp. 151--178, 1998.
```
Original content found in: `[37] Herbster1998_TrackingBestExpert.pdf` pag. 151
Citation verification details: Section 1 and 3 on pages 151–157 introduce the Fixed-Share algorithm for tracking shifting best experts in online prediction; matches Machine Learning (Kluwer), vol. 32, no. 2, pp. 151–178, 1998.

---

[38] UNVERIFIED!!!
Paper says: "...as in system identification with a known input signal [38]." Pag. 7
Paper citation says:
```
[38] L. Ljung, System Identification: Theory for the User, 2nd ed. Upper Saddle River, NJ: Prentice Hall, 1999.
```
Original content found in: `None (Missing from references folder)`
Citation verification details: The PDF is unavailable in references/ because the publication is a 500-page commercial textbook (Lennart Ljung, 'System Identification: Theory for the User', 2nd ed., Prentice Hall, 1999, ISBN 978-0136566953).

---

[39] UNVERIFIED!!!
Paper says: "Probing an environment to infer its dynamics has also been learned in RL [39]." Pag. 7
Paper citation says:
```
[39] W. Zhou, L. Pinto, and A. Gupta, "Environment probing interaction policies," in International Conference on Learning Representations, 2019.
```
Original content found in: `[39] Zhou2019_EnvironmentAdaptation.pdf`
Citation verification details: The file stored in references/ is an unrelated mathematics preprint on spectral triples (arXiv:1903.07599). The true paper is 'Environment probing interaction policies' by Wenxuan Zhou, Lerrel Pinto & Abhinav Gupta in ICLR 2019.

---

[40] VERIFIED.
Paper says: "Each observation channel is decomposed into a Haar wavelet packet tree of depth 3 [40]..." Pag. 7
Paper citation says:
```
[40] R. R. Coifman and M. V. Wickerhauser, "Entropy-based algorithms for best basis selection," IEEE Transactions on Information Theory, vol. 38, no. 2, pp. 713--718, 1992.
```
Original content found in: `[40] Coifman1992_Entropy.pdf` pag. 1
Citation verification details: Pages 1–3 define best-basis selection and entropy-based criteria on wavelet packet trees; matches IEEE Trans. Information Theory, vol. 38, no. 2, pp. 713–718, 1992.

---

[41] VERIFIED.
Paper says: "Like the discrete wavelet transform [41], the tree is dyadic and non-overlapping, but it splits every band." Pag. 7
Paper citation says:
```
[41] S. G. Mallat, "A theory for multiresolution signal decomposition: The wavelet representation," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 11, no. 7, pp. 674--693, 1989.
```
Original content found in: `[41] Mallat1989_WaveletTheory.pdf` pag. 674
Citation verification details: Section I and II on pages 674–679 formulate multiresolution signal decomposition and the dyadic discrete wavelet representation; matches IEEE TPAMI, vol. 11, no. 7, pp. 674–693, 1989.

---

[42] UNVERIFIED!!!
Paper says: "...standard signature of a change of physics, in structural damage detection and machine fault diagnosis [42]." Pag. 7
Paper citation says:
```
[42] R. Yan, R. X. Gao, and X. Chen, "Wavelets for fault diagnosis of rotary machines: {A} review with applications," Signal Processing, vol. 96, pp. 1--15, 2014.
```
Original content found in: `None (Missing from references folder)`
Citation verification details: The PDF is unavailable in references/ because the publication is behind Elsevier paywall ('Wavelets for fault diagnosis of rotary machines: A review with applications', Signal Processing, vol. 96, pp. 1–15, 2014, DOI: 10.1016/j.sigpro.2013.04.015).

---

[43] VERIFIED.
Paper says: "The transforms are computed with PyWavelets [43]." Pag. 7
Paper citation says:
```
[43] G. R. Lee, R. Gommers, F. Waselewski, K. Wohlfahrt, and A. O'Leary, "{PyWavelets}: A {Python} package for wavelet analysis," Journal of Open Source Software, vol. 4, no. 36, p. 1237, 2019.
```
Original content found in: `[43] Lee2019_PyWavelets.pdf` pag. 1
Citation verification details: Page 1 presents PyWavelets as an open-source Python package for continuous, discrete, and wavelet packet transforms; matches Journal of Open Source Software (JOSS), vol. 4, no. 36, p. 1237, 2019.

---

[44] VERIFIED.
Paper says: "ENA was evaluated in two non-stationary domains built on Gymnasium [44]: the discrete-action CartPole-v1 [11], and the continuous-action HalfCheetah-v4 [12]." Pag. 8
Paper citation says:
```
[44] M. Towers, A. Kwiatkowski, J. Terry, J. U. Balis, G. De Cola, T. Deleu, M. Goul{\ a}o, A. Kallinteris, M. Krimmel, A. KG, R. Perez-Vicente, A. Pierr{\'e}, S. Schulhoff, J. J. Tai, H. Tan, and O. G. Younis, "Gymnasium: A standard interface for reinforcement learning environments," arXiv preprint arXiv:2407.17032, 2024.
```
Original content found in: `[44] Towers2024_Gymnasium.pdf` pag. 1
Citation verification details: Page 1 introduces Gymnasium as the standardized API and benchmark repository successor to OpenAI Gym; matches Towers et al., arXiv:2407.17032, 2024.

---

[45] VERIFIED.
Paper says: "...ENA-Combined was compared with DQN [13], PPO [14] and the discreteaction variant of SAC (SAC-Discrete) [45]." Pag. 9
Paper citation says:
```
[45] P. Christodoulou, "Soft actor-critic for discrete action settings," arXiv preprint arXiv:1910.07207, 2019.
```
Original content found in: `[45] Christodoulou2019_SACDiscrete.pdf` pag. 1
Citation verification details: Section 3 on pages 2–3 derives the discrete-action formulation of Soft Actor-Critic (SAC-Discrete); matches Christodoulou, arXiv:1910.07207, 2019.

---

[46] VERIFIED.
Paper says: "The Haar and db4 [46] trains depend strongly on the features." Pag. 14
Paper citation says:
```
[46] I. Daubechies, "Orthonormal bases of compactly supported wavelets," Communications on Pure and Applied Mathematics, vol. 41, no. 7, pp. 909--996, 1988.
```
Original content found in: `[46] Daubechies1988_OrthonormalBases.pdf` pag. 909
Citation verification details: Page 909 and Table 1 construct compactly supported orthonormal wavelets, including the db4 (Daubechies 4) filter coefficients; matches Comm. Pure Appl. Math., vol. 41, no. 7, pp. 909–996, 1988.

---

[47] UNVERIFIED!!!
Paper says: "The cost of exploring in parameter space grows with the number of parameters [47]..." Pag. 17
Paper citation says:
```
[47] A. Vemula, W. Sun, and J. A. Bagnell, "Contrasting exploration in parameter and action space: A zeroth-order optimization perspective," in Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics (AISTATS), ser. Proceedings of Machine Learning Research, vol. 89, 2019, pp. 2926--2935.
```
Original content found in: `[47] Vemula2019_Contrasting.pdf`
Citation verification details: The file stored in references/ is an unrelated information theory preprint (Aliasgari et al., arXiv:1901.07705). The true paper is 'Contrasting Exploration in Parameter and Action Space: A Zeroth-Order Optimization Perspective' by Anirudh Vemula, Wen Sun & J. Andrew Bagnell in AISTATS 2019 (PMLR v89, pp. 2926–2935).

---

[48] VERIFIED.
Paper says: "...and random mutations tend to disrupt large networks [48]. This is why ENA’s specialists are small." Pag. 17
Paper citation says:
```
[48] J. Lehman, J. Chen, J. Clune, and K. O. Stanley, "Safe mutations for deep and recurrent neural networks through output gradients," in Proceedings of the Genetic and Evolutionary Computation Conference (GECCO), 2018.
```
Original content found in: `[48] Lehman2018_SafeMutations.pdf` pag. 1
Citation verification details: Section 1 on page 1 analyzes catastrophic phenotypic disruption from random mutations in large neural networks and introduces safe mutations via output gradients; matches GECCO 2018, pp. 117–124.

---

[49] UNVERIFIED!!!
Paper says: "On HalfCheetah, PPO and SAC take the settings of RL Baselines3 Zoo [49] (Table XI)." Pag. 20
Paper citation says:
```
[49] A. Raffin, "{RL} baselines3 zoo," https://github.com/DLR-RM/rl-baselines3-zoo, 2020. \end{thebibliography}
```
Original content found in: `[49] Raffin2020_RLZoo.pdf`
Citation verification details: The file stored in references/ is a paper by Jens Kober on 'Smooth Exploration for Robotic RL'. The cited entry is Antonin Raffin's RL Baselines3 Zoo code repository and documentation (https://github.com/DLR-RM/rl-baselines3-zoo, 2020).

---

