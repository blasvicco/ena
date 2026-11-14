# Reference Download Status

**47 PDFs downloaded, plus 1 code repository checked online** (48 of 51 citations, 94.1% coverage). Updated 2026-11-14, after the citation fixes in `docs/citation_fix_plan.md` were applied. Those fixes added [36] and [38], which renumbered every later reference.
- 47 citations have their source PDF in `references/`. "Downloaded" means the file is the cited work. Whether the paper uses each source correctly is tracked separately in `docs/citations.md`.
- 1 citation ([36]) is a code repository, so there is no PDF. It was checked online.
- 3 citations are still unavailable (a paywalled journal article, a paywalled book chapter, a commercial textbook).
- File names follow `[NN] AuthorYear_ShortTitle.pdf`, where `NN` is the reference number in the current build (`app/latex/build/main.bbl`). Adding or removing a citation renumbers the later references, so rename the files to match.

---

## ©️ Copyright and git

The repository is public, so only PDFs whose license allows redistribution are tracked in git. All other PDFs are kept locally for the citation audit and ignored by `.gitignore` (`references/**/*.pdf`, with an allow-list matched by author and year). Being free to read online is not enough. The license must allow others to share the file. arXiv's default license (`arxiv.org/licenses/nonexclusive-distrib/1.0`) only lets arXiv itself distribute the paper. Licenses checked 2026-11-14: arXiv versions through arXiv's metadata service (oaipmh.arxiv.org), other files from the notice printed in the PDF.

Tracked in git (6):

| # | File | License |
|---|------|---------|
| [21] | `[21] Yazdani2021_SurveyDynamicOptimization.pdf` | CC BY (stated on the repository cover page) |
| [24] | `[24] Lillo2026_BeyondSingleTask.pdf` | CC BY 4.0 (arXiv:2604.15414) |
| [29] | `[29] Cobb1990_Investigation.pdf` | US Government work: "Approved for public release; distribution unlimited" |
| [45] | `[45] Lee2019_PyWavelets.pdf` | CC BY (JOSS) |
| [46] | `[46] Towers2024_Gymnasium.pdf` | CC BY 4.0 (arXiv:2407.17032) |
| [51] | `[51] Raffin2020_RLZoo.pdf` | CC BY 4.0 (JMLR) |

Local only (41): everything else, including `to_classify/`. They have either the arXiv default license (19, including [26], which predates arXiv's license field), a publisher or society copyright notice, or no license at all, which means all rights reserved. [05] Kessler is CC BY 4.0 on arXiv, but the PDF itself says "Copyright © 2022, AAAI... All rights reserved", so it is kept local to be safe.

To add a new PDF to git, check its license first, then add an allow-list line to `.gitignore`.

---

## ✅ Downloaded (47)

| # | File | Source / Method |
|---|------|-----------------|
| [01] | `[01] McCloskey1989_CatastrophicInterference.pdf` | Psychology of Learning and Motivation vol. 24, pp. 109–165 / added manually 2026-11-14 |
| [02] | `[02] Khetarpal2022_ContinualRL.pdf` | JAIR vol. 75 / arXiv:2012.13490 |
| [03] | `[03] Kirkpatrick2017_OvercomingCatastrophicForgetting.pdf` | PNAS vol. 114 / arXiv:1612.00796 |
| [04] | `[04] Rolnick2019_ExperienceReplay.pdf` | NeurIPS vol. 32 / arXiv:1811.11682 |
| [05] | `[05] Kessler2022_SameEnvironment.pdf` | AAAI vol. 36 / arXiv:2106.02940 |
| [06] | `[06] daSilva2006_NonStationaryRL.pdf` | ICML 2006 / UMass CS |
| [07] | `[07] Alegre2021_MinimumWidth.pdf` | AAMAS 2021 / arXiv:2105.09452 |
| [08] | `[08] Dick2024_Statistical.pdf` | CoLLAs 2024 / PMLR vol. 274 |
| [09] | `[09] Nisioti2026_ContinualQD.pdf` | arXiv:2610.01583 |
| [10] | `[10] Rosman2016_BayesianPolicyReuse.pdf` | Machine Learning (Springer) / arXiv:1505.00284 |
| [11] | `[11] Barto1983_NeuronLike.pdf` | IEEE Trans. SMC vol. 13 / Washington CS (scanned, no text layer) |
| [12] | `[12] Todorov2012_MuJoCo.pdf` | IEEE IROS 2012 / UW CS archive |
| [13] | `[13] Mnih2015_HumanLevelDQN.pdf` | Nature vol. 518 / DeepMind official mirror |
| [14] | `[14] Schulman2017_PPO.pdf` | arXiv:1707.06347 |
| [15] | `[15] Haarnoja2018_SAC.pdf` | ICML 2018 / arXiv:1801.01290 |
| [16] | `[16] Stanley2019_DesigningNeuralNetworks.pdf` | Nature Machine Intelligence vol. 1, pp. 24–35 / added manually 2026-11-14 |
| [17] | `[17] Salimans2017_EvolutionStrategies.pdf` | arXiv:1703.03864 |
| [18] | `[18] Such2017_DeepNeuroevolution.pdf` | arXiv:1712.06567 |
| [19] | `[19] Khadka2018_EvolutionGuidedPolicyGradient.pdf` | NeurIPS vol. 31 / arXiv:1805.07917 |
| [20] | `[20] Pourchot2019_CEMRL.pdf` | ICLR 2019 / arXiv:1810.01222 |
| [21] | `[21] Yazdani2021_SurveyDynamicOptimization.pdf` | IEEE TEVC vol. 25 / Aberystwyth Pure repo |
| [22] | `[22] Mouret2015_IlluminatingSearch.pdf` | arXiv:1504.04909 |
| [23] | `[23] Cully2015_RobotsAdapt.pdf` | Nature vol. 521 / arXiv:1407.3501 |
| [24] | `[24] Lillo2026_BeyondSingleTask.pdf` | CoLLAs 2026 / arXiv:2604.15414 |
| [25] | `[25] Page1954_ContinuousSampling.pdf` | *Biometrika* vol. 41 (1954) |
| [26] | `[26] Adams2007_BayesianChangepoint.pdf` | arXiv:0710.3742 |
| [27] | `[27] Halme2026_SequentialChangepoint.pdf` | arXiv:2606.05072 |
| [28] | `[28] Galhardo2007_MutationStressResponse.pdf` | Crit. Rev. Biochem. Mol. Biol. vol. 42, pp. 399–435 / added manually 2026-11-14 |
| [29] | `[29] Cobb1990_Investigation.pdf` | NRL Memorandum Report 6760 (DTIC ADA229159) |
| [32] | `[32] McClelland1995_ComplementaryLearning.pdf` | Psychological Review vol. 102 / Stanford repo |
| [33] | `[33] Kumaran2016_ComplementaryLearningSystems.pdf` | Trends in Cognitive Sciences vol. 20 / Stanford repo |
| [34] | `[34] Rosin1997_NewMethods.pdf` | Evolutionary Computation vol. 5 / CUNY ALife repo (UCSD TR CS96-491 preprint; no text layer) |
| [35] | `[35] Branke1999_Memory.pdf` | IEEE CEC 1999 |
| [37] | `[37] Mamdani1975_FuzzyLogic.pdf` | Int. J. Man-Machine Studies vol. 7 (1975) |
| [38] | `[38] Lee1990_FuzzyLogicControllerPartII.pdf` | IEEE Trans. SMC vol. 20, no. 2, pp. 419–435 (1990), doi:10.1109/21.52552 / added manually 2026-11-14 |
| [39] | `[39] Herbster1998_TrackingBestExpert.pdf` | Machine Learning vol. 32 / Warmuth repo |
| [41] | `[41] Zhou2019_EnvironmentAdaptation.pdf` | ICLR 2019 / arXiv:1907.11740 |
| [42] | `[42] Coifman1992_Entropy.pdf` | IEEE Trans. IT vol. 38 / WUSTL Math repo (preprint; no text layer) |
| [43] | `[43] Mallat1989_WaveletTheory.pdf` | IEEE TPAMI vol. 11 / ENS faculty repo |
| [44] | `[44] Yan2014_WaveletsFaultDiagnosis.pdf` | Signal Processing vol. 96, pp. 1–15 / added manually 2026-11-14 |
| [45] | `[45] Lee2019_PyWavelets.pdf` | JOSS vol. 4 (2019) |
| [46] | `[46] Towers2024_Gymnasium.pdf` | arXiv:2407.17032 |
| [47] | `[47] Christodoulou2019_SACDiscrete.pdf` | arXiv:1910.07207 |
| [48] | `[48] Daubechies1988_OrthonormalBases.pdf` | Comm. Pure Appl. Math. vol. 41 / Duke Math repo |
| [49] | `[49] Vemula2019_Contrasting.pdf` | AISTATS 2019 / arXiv:1901.11503 |
| [50] | `[50] Lehman2018_SafeMutations.pdf` | GECCO 2018 / arXiv:1712.06563 |
| [51] | `[51] Raffin2020_RLZoo.pdf` | JMLR vol. 22 (2021), the Stable-Baselines3 paper. The cited work is the RL Baselines3 Zoo GitHub repository, which this paper cites. |

## 🌐 Checked online, no PDF (1)

| # | Source | What was checked |
|---|--------|------------------|
| [36] | OpenAI, evolution-strategies-starter, https://github.com/openai/evolution-strategies-starter (archived, read-only since Apr 2026) | `es_distributed/policies.py`: hidden layers use `normc_initializer(1.0)`, and the continuous output is a linear `normc_initializer(0.01)` layer. Checked 2026-11-14. |

---

## ❌ Remaining Missing (3)

| # | Citation | Type & Reason |
|---|----------|---------------|
| [30] | Karafotias et al. 2015 | Journal article in *IEEE Transactions on Evolutionary Computation* (vol. 19, pp. 167–187). Paywalled. |
| [31] | Grefenstette 1992 | Book chapter in *Parallel Problem Solving from Nature 2*, North-Holland/Elsevier. Paywalled. Confirm the page range (bib: pp. 137–144). |
| [40] | Ljung 1999 | 500-page Prentice Hall commercial textbook (*System Identification: Theory for the User*). |

---

## 🗂 `to_classify/`

Neither file is cited by the paper, and neither matches a missing reference. Keep them or delete them.

| File | What it is |
|------|------------|
| `function-optimization-in-nonstationary-environment-using-steady-.pdf` | A. Ghosh, S. Tsutsui, H. Tanaka, "Function optimization in nonstationary environment using steady state genetic algorithms with aging of individuals" (IEEE 1998 proceedings, ISBN 0-7803-4869-9, likely ICEC 1998) |
| `robinson1975.pdf` | G. K. Robinson, "Some counterexamples to the theory of confidence intervals," *Biometrika* 62(1), 1975 |
