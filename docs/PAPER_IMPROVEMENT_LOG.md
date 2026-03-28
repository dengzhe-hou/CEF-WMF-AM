# Paper Improvement Log — cef_paper_v7.tex

## Target: NeurIPS 2026 Main Track (Position Paper)

## Score Progression

| Round | Score | Verdict | Key Changes |
|-------|-------|---------|-------------|
| Loop 1, Round 0 | 5/10 | No | Baseline — compiled first time with tectonic |
| Loop 1, Round 1→2 | 5→6/10 | No→Almost | Ceiling defense, composites labeled exploratory, LOMO, construct mapping |
| Loop 2, Round 1→2 | 6→7/10 | Almost→YES | Reframed as construct-development, tightened bridge, falsification criteria |
| Loop 3, Round 0 | — | — | Integrated WMF-AM control task + MCC-CE-v2 results into paper |
| Loop 3, Round 1 | 5/10 | Almost | Validity not self-contained, terminology overreach, MCC-CE-v2 overclaimed, control shortcut |
| Loop 3, Round 2 | 6/10 | Almost | All Round 1 issues fixed; global terminology + rubric appendix added |
| Loop 4, Round 1 | 5/10 | Almost | Claims > evidence, construct validity incomplete, statistical language too strong |
| Loop 4, Round 2 | 6/10 | Almost | Narrowed framing, rival explanations, evidence map, CLA-DC collapsed |
| Loop 5, Round 1 | 6/10 | Almost | Framing overreach, missing theory→framework argument, metaphor inflation, "validated" overuse |
| Loop 5, Round 2 | **7/10** | **YES** | Scope compression, Goodhart citation, low-status probe prose trimmed |
| Loop 6, Round 1 | 6/10 | Almost | **NeurIPS format restructure: 19→8 pages main body** + appendix |
| Loop 6, Round 2 | 6/10 | Almost | Thesis/framework separation, pruned Sec 3-4, appendix dedup |
| Loop 7, Round 1 | 5/10 | Almost | tau fix, 16x removed, formal defs, methods table, CLA-DC collapse |
| Loop 7, Round 2 | 6/10 | Almost | CI-based dissociation, ε/δ robustness, control weakened, AGENT-PQ defined |
| Loop 8, Round 0 | — | — | **N=7→N=15**: integrated 8 expansion models + yoked cancellation control |
| Loop 8, Round 1 | 5/10 | Almost | Reframed dissociation, K-depth fix, completion caveat, alt explanations |
| Loop 8, Round 2 | 6/10 | Almost | 0 CRITICALs; 5 MAJORs remain (thin empirical base, template confound) |
| Loop 9, Round 0 | — | — | **3 new experiments**: 100-item battery, 4-seed all 15 models, template harmonization |
| Loop 9, Round 1 | 6/10 | Almost | Reframed "dissociation"→"differentiation", power analysis, permutation test, range restriction |
| Loop 9, Round 2 | **7/10** | **YES** | Claims-evidence alignment strong; predictive validity deferred explicitly |
| Loop 10, Round 0 | — | — | **N=15→N=20**: integrated predictive validity (τ=0.612), 5 small models, agent validation |
| Loop 10, Round 1 | 5/10 | No | Dissociation overclaimed, predictive validity under-controlled, not self-contained, stats underspecified |
| Loop 10, Round 2 | 6/10 | Almost | Added param control (partial τ=0.503), within-Qwen (τ=0.894), LOO [0.537,0.624], task split (non-WMF τ=0.623), stats paragraph, worked examples |
| Loop 11, Round 1 | 6/10 | Almost | Formalization ad hoc, claims too broad, construct validity incomplete, not self-contained, notation inconsistent |
| Loop 11, Round 2 | **7/10** | **YES** | Benchmark-conditional formalization, existence-proof framing, AGENT-PQ/CONV-RFPOC clarified, p-values standardized, ablation status explicit |
| Loop 12, Round 1 | 7/10 | Almost | Non-arith + paraphrase ablations integrated; CEF hierarchy sharpened; downstream claim narrowed to "10-task battery" |
| Loop 12, Round 2 | **8/10** | **YES** | Arithmetic accumulation vs assignment explained; standalone arithmetic limitation explicitly acknowledged; scoped claims throughout |

## Loop 11 — Claim Calibration + Self-Containedness (Mar 17, 2026)

### Loop 11 Round 1 Review (GPT-5.4 xhigh)

**Score: 6/10 (Almost)**

Strengths:
- Core problem statement compelling and relevant
- Three-layer claim separation unusually careful
- Pilot more rigorous than many benchmark proposals (seeds, templates, yoked, partial τ, LOO)
- Framework operational rather than purely rhetorical
- Honest about limitations

Weaknesses (5 MAJORs):
1. MAJOR: Formalization too contingent — ad hoc thresholds (τ≥0.8, <0.55 at 80% power) feel arbitrary
2. MAJOR: Claims-evidence misalignment — title/abstract imply general finding, evidence is one probe/one downstream suite
3. MAJOR: WMF-AM construct validity incomplete — prompt sensitivity, instruction-following, tokenizer confounds not fully ruled out
4. MAJOR: Not self-contained — AGENT-PQ, CONV-RFPOC unclear; Phase 6 scoring inconsistency (exact-match vs LLM judge)
5. MAJOR: Notation/reporting inconsistency — mixed p-value formats, repeated headline stats

### Fixes Implemented
1. Formalization recast as benchmark-conditional: insufficiency defined relative to ⟨Y, Z, D⟩ triple; removed ad hoc thresholds
2. Title narrowed with subtitle; abstract explicitly frames as existence proof
3. Phase 6 scoring clarified: deterministic verification (exact match) for Agent score; AGENT-PQ (LLM-judged) separated for convergent validity only
4. CONV-RFPOC defined inline on first use
5. All p=0.0003 → p<0.001; consistent formatting throughout
6. Validity summary text deduplicated — references table instead of restating numbers
7. Ablation status enumerated: completed (3 surface forms, 3 templates, 4 seeds) + planned (paraphrase, non-arithmetic, instruction-following baseline)
8. Added Nye et al. 2021 (scratchpads) citation
9. Related work: explicit distinction from process supervision and mechanistic interpretability
10. Glossary forward-referenced in introduction

### Loop 11 Round 2 Review (GPT-5.4 xhigh)

**Score: 7/10 (Yes)**

Strengths:
- Claims-evidence alignment now much stronger
- Formalization more defensible (benchmark-conditional)
- Self-containedness substantially better
- Empirical pilot unusually careful for position paper
- Related-work positioning sharper

Remaining MAJORs (both require new experiments, not writing fixes):
1. WMF-AM construct validity still needs targeted ablations (non-arithmetic tracking, prompt paraphrase, instruction-following baseline)
2. External validity narrow — all open-weight; needs cloud API sanity check or second downstream benchmark

### Minor Round 2 Fixes Applied
- Softened "working memory" labeling → "state-tracking probe inspired by working-memory paradigms"
- Added sensitivity check for dissociation-pair thresholds (18/105 at stricter ≤0.03/≥0.15)

## Loop 7 — Statistical Rigor + Self-Containedness (Mar 17, 2026)

### Loop 7 Round 1 Review (GPT-5.4 xhigh)

**Score: 5/10 (Almost)**

Strengths:
- Core thesis strong and relevant
- WMF-AM pilot more rigorous than typical (multi-seed, control)
- Explicit scope separation (thesis, CEF, WMF-AM)
- Unusually honest limitations

Weaknesses:
- CRITICAL: Empirical evidence supports only narrow slice; CEF framing still broader than evidence
- CRITICAL: 16x headline partly induced by ceiling matching; tau inconsistency (0.07 vs 0.206)
- MAJOR: "Formalization" not actually formal; no measurement model
- MAJOR: Main body not self-contained (missing methods details)
- MAJOR: Construct isolation incomplete (2 models degrade on control)
- MAJOR: CLA-DC collapse threatens four-dimension structure
- MINOR: Repetitive caveats; safety/OOD motivation speculative

Fixes:
1. Fixed tau inconsistency: all instances now 0.206 (p=0.530)
2. Removed ALL "16x" framing; replaced with matched-pair counts (14/21 pairs)
3. Added "Formal definitions" paragraph: insufficiency, dissociation, falsification criteria
4. Added compact methods table (Table: Pilot design summary)
5. Rewrote CLA paragraph: CLA-DC provisionally collapsed onto WMF; CLA-RA/CLA-CR independent by design
6. Added Jacobs & Wallach (2021) citation
7. Added ceiling-artifact checks (MMLU/GSM8K correlations + distribution-free matched-pair count)

### Loop 7 Round 2 Review (GPT-5.4 xhigh)

**Score: 6/10 (Almost)**

Strengths:
- Reframing much better: clean thesis/CEF/WMF-AM separation
- Matched-pair counts less rhetorically inflated than 16x
- Tau consistency fixed
- Formal definitions help falsifiability
- Methods table improves self-containedness
- CLA-DC rewrite sensible

Weaknesses:
- No CRITICALs remaining
- MAJOR: Dissociation criterion still weak (null-failure at N=7)
- MAJOR: ε/δ thresholds ad hoc
- MAJOR: Control design still not fully isolated
- MAJOR: Framework-level claims depend on future N≥15 work
- MAJOR: AGENT-PQ, CONV-RFPOC under-defined in main text
- MINOR: CLA collapse should be reflected consistently in tables/figures
- MINOR: Caveat repetition; confirmatory vs exploratory labeling

Fixes:
1. Replaced null-failure dissociation with CI-based: "tau=0.206, bootstrap 95% CI [-0.467, 0.733]"
2. Added ε/δ robustness: δ=0.50 → 10/21 pairs; ε=0.02 → 6/21 pairs
3. Weakened control language to "partial construct support, not isolation"
4. Added inline definitions for AGENT-PQ (GPT-4o judge, 1-5 rubric) and CONV-RFPOC (rank-fusion composite)

### PDFs
- `cef_paper_v7_loop7_round0.pdf` — Baseline (same as loop6_round2)
- `cef_paper_v7_loop7_round1.pdf` — After Round 1
- `cef_paper_v7_loop7_round2.pdf` — Final (16 pages, ~8 main body)

---

## Loop 6 — NeurIPS Format Compliance: 19→8 Pages Main Body (Mar 17, 2026)

### Loop 6 Round 1 Review (GPT-5.4 xhigh)

**Score: 6/10 (Almost)**

Focus: Paper was 19 pages main body, NeurIPS limit is 9. Reviewer provided format compliance plan.

Fixes:
1. **Restructured entire paper**: 19 pages → 8 pages main body + expanded appendix
2. Merged "Completion Fallacy" + "Background/Related Work" into one section
3. Merged "Limitations" + "Discussion" into one section
4. CEF section: kept framework figure, dimension table, WMF-AM flagship, MCC/EMC/CLA brief; moved sub-dim details to appendix
5. Pilot section: kept dissociation figure+table, depth figure, control table, validity summary; moved study matrix, EMC-lite table, full validity to appendix
6. 17 appendix sections covering all moved content
7. Fixed overfull hbox warnings (28pt → <10pt)

### Loop 6 Round 2 Review (GPT-5.4 xhigh)

**Score: 6/10 (Almost)**

Strengths:
- Main-body arc now coherent: thesis, evidence, framework, pilot, discussion
- Claim calibration good
- Unusually substantive for position paper
- Contributes usable research agenda

Weaknesses:
- MAJOR: Thesis still entangled with framework; skeptic may think claim depends on WMF-AM
- MAJOR: Sections 3-4 still overcompressed; too many named constructs for main body
- MAJOR: Some overclaiming relative to evidence
- MINOR: Appendix has duplicate tables
- MINOR: Acronym load high

Fixes:
1. Added "Three separable claims" paragraph in Introduction: general thesis, CEF as framework, WMF-AM as operationalization
2. Added "Separating thesis from framework" paragraph in Discussion
3. Moved at-a-glance table, study matrix, EMC-lite table to appendix (kept synthesis sentences)
4. Tightened "fallacy" language: "completion alone is insufficient for inferring latent process competence"
5. Demoted architectural observation to "exploratory"
6. Removed duplicate tables in appendix (control, validity)
7. Fixed overfull hbox (18pt → 0.3pt)

### PDFs
- `cef_paper_v7_loop6_round0.pdf` — Original (19 pages main body)
- `cef_paper_v7_loop6_round1.pdf` — After Round 1 (8 pages main body)
- `cef_paper_v7_loop6_round2.pdf` — Final (7.5 pages main body, NeurIPS compliant)

---

## Loop 5 — Research Agenda Reframing + Language Discipline (Mar 17, 2026)

### Loop 5 Round 1 Review (GPT-5.4 xhigh)

**Score: 6/10 (Almost)**

Weaknesses:
- CRITICAL: Framing overreach — Section 4 opens as if CEF is established, not proposed
- MAJOR: Missing theory-to-framework argument — jumps from "completion is insufficient" to 11 sub-dimensions
- MAJOR: Metaphor inflation — cognitive-science labels imply mechanism claims
- MAJOR: "Validated" language applied throughout despite only WMF-AM having multi-evidence support

Fixes:
1. Added "CEF is a proposed research agenda—a taxonomy of process-sensitive probes..." opening to Section 4
2. Updated Figure 3 caption: "Solid border: pilot-supported (WMF-AM). Dashed border: preliminary or proposed probes"
3. Added new subsection "From Outcome Underdetermination to Process Probes" (Section 2.1) with three-part argument
4. Added "On cognitive-science labels" paragraph: labels are operational shorthand, not mechanism claims
5. Systematically restricted "validated/validation" (~20 instances): "pilot-validated" → "pilot-supported", preserved only for human paradigm references

### Loop 5 Round 2 Review (GPT-5.4 xhigh)

**Score: 7/10 (Accept — YES)**

Strengths:
- Round 1 weaknesses fixed substantively, not cosmetically
- Claims-vs-evidence alignment now strong
- New Section 2.1 closes important theoretical gap
- Cognitive-science labeling well disciplined
- Unusually honest reporting throughout
- Paper architecture mature (glossary, evidence map, status table, confounds/falsifiability)

Remaining weaknesses:
- MAJOR: Framework surface area still larger than evidence can animate (11 sub-dimensions in main text)
- MINOR: Acronym/taxonomy density still high
- MINOR: Section 2.1 risks conceptual repetition with intro and Section 4
- MINOR: Statistical detail may project more precision than N=7 supports
- MINOR: "Fallacy" in title slightly stronger than the nuanced stance

Fixes:
1. Compressed EMC sub-dimension prose (removed verbose descriptions, added status cross-reference)
2. Compressed CLA sub-dimension prose (shortened CLA-RA/CLA-CR descriptions, added status note)
3. Shortened CLA "Core insight" paragraph from 8 lines to 4
4. Added Goodhart's law citation to Completion Fallacy definition
5. All missing references already cited (Cronbach & Meehl 1955, Messick 1989, Ribeiro 2020)

## Loop 4 — Framing Tightening + Evidence Boundary (Mar 17, 2026)

### Loop 4 Round 1 Review (GPT-5.4 xhigh)

**Score: 5/10 (Almost)**

Weaknesses:
- CRITICAL: Paper's main claim stronger than evidence; broad framework from narrow pilot
- CRITICAL: Construct validity not strong enough; probes may measure prompt sensitivity, not cognitive constructs
- MAJOR: Statistical language too strong for N=7 ("confirm robustness")
- MAJOR: Dissociation partly engineered by ceiling compression; "16× larger" framing
- MAJOR: Self-containedness: 7-vs-8 models, τ=0.070 vs τ=0.206 discrepancy
- MAJOR: WMF-AM/CLA-DC τ=0.905 threatens dimensional separation
- MAJOR: AGENT-PQ convergent validity may reflect shared LLM-judge bias
- MINOR: Notation burden too high
- MINOR: Prose slips toward direct cognitive measurement language

Fixes:
1. Softened "every benchmark should" → "benchmarks would benefit from"
2. Added "Rival explanations" paragraph with explicit confounds for WMF-AM, MCC, EMC
3. All "confirm" → "suggest"/"indicate"/"support" (12 instances)
4. Added "16× should be read as existence proof, not general effect size"
5. Clarified N=7 vs N=8 (llama3.1:70b added for MCC-CE-v2 only); disambiguated τ=0.070 vs τ=0.206
6. CLA-DC: "no evidence these are separable; provisionally collapsed"
7. Added AGENT-PQ LLM-judge limitation paragraph
8. Added Glossary table (Table 1) with all acronyms
9. Added missing refs: Turpin 2023, Lanham 2023, Uesato 2022
10. Added Study Matrix table (Table 3)
11. Wrapped valstatus table in resizebox (was 87pt overfull)

### Loop 4 Round 2 Review (GPT-5.4 xhigh)

**Score: 6/10 (Borderline weak accept)**

Remaining weaknesses:
- MAJOR: Still spans more conceptual territory than evidence can carry
- MAJOR: Construct validity improved but still incomplete without ablations
- MAJOR: May feel split between position paper and benchmark paper genres
- MAJOR: Statistical interpretation still fragile at N=7
- MINOR: CLA-DC "provisionally collapsed" but still named dimension
- MINOR: 11 subdimension inventory may dominate main text
- MINOR: Recommendations still slightly ahead of evidence

Fixes:
1. Reframed contribution (2) to make WMF-AM the "flagship validated pilot"
2. Added explicit Evidence Map: pilot-validated / preliminary / suggestive / proposed
3. Added three-level evidence structure in validity section (supported / suggested / requires N≥15)
4. Changed "Recommendations" → "Research agenda" with softer language
5. Added "Why retain the framework despite collapse?" paragraph
6. De-emphasized incremental validity: "not inferentially meaningful; effect direction only"

## All PDFs
- `cef_paper_v7_round0_original.pdf` — Before Loop 3
- `cef_paper_v7_round1.pdf` — After Loop 3 Round 1
- `cef_paper_v7_round2.pdf` — After Loop 3 Round 2
- `cef_paper_v7_loop4_round0.pdf` — Before Loop 4
- `cef_paper_v7_loop4_round1.pdf` — After Loop 4 Round 1
- `cef_paper_v7_loop4_round2.pdf` — After Loop 4 Round 2
- `cef_paper_v7_loop5_round0.pdf` — Before Loop 5
- `cef_paper_v7_loop5_round1.pdf` — After Loop 5 Round 1
- `cef_paper_v7_loop5_round2.pdf` — After Loop 5 Round 2 (current)

## ThreadIds
- Loop 2: `019cf75b-99ca-7971-a539-c23213c05e7b`
- Loop 3: `019cf78b-b9f7-7f42-b2e1-329d7fabd46c`
- Loop 4: `019cf7a1-59e8-78e2-b0c4-fcfb5adeb502`
- Loop 5: `019cf7bd-d17a-7fb2-ab6c-51afec2a2bff`
- Loop 9: `019cfb88-03a2-76f0-983a-762a44b6c488`

## Loop 9 — Full Experiment Integration + Claims Calibration (Mar 17, 2026)

**Major changes**: Integrated 3 new experiments (100-item completion battery, 4-seed replication for all 15 models, template harmonization). Reframed from "dissociation" to "existence-proof differentiation". Added power analysis, permutation tests, range-restriction caveats.

### Loop 9 Round 0 (Pre-review edits)
- Abstract: Rewritten with 100-item battery, 5× range disparity framing
- Statistics: τ=0.279 (p=0.150) with new battery; τ(WMF-Yoked, Outcome)=0.125 (p=0.519)
- Template harmonization: τ(bare,chat)=0.524 (p=0.006) — strongest significant result
- All 15 models now have 4-seed WMF-AM (expansion 8 previously had 1 seed)
- Pilot table updated with 100-item Outcome scores and 4-seed WMF-AM means
- Methods table: added Phase 5 (Template Harmonization)
- Figures: fig1_dissociation.pdf and fig2_depth_profile.pdf regenerated for N=15

### Loop 9 Round 1 Review (GPT-5.4 xhigh)

**Score: 6/10 (Almost)**

Strengths:
- Claims-evidence alignment better than most position papers
- Flagship pilot reasonably credible: N=15, 100 items, 4 seeds, yoked control, template harmonization
- Yoked cancellation control is strong design
- Careful about labels being operational, not mechanistic

Weaknesses:
- CRITICAL: τ=0.279, p=0.150, CI [-0.156, 0.637] too weak for "dissociation" framing
- MAJOR: No predictive validity on downstream criterion
- MAJOR: Range restriction (0.75-0.92) may explain low τ
- MAJOR: Four-dimension taxonomy under-validated
- MAJOR: Construct mapping not tight enough

Fixes:
1. Reframed "dissociation" → "existence-proof differentiation" throughout
2. Added power analysis (min detectable τ=0.55 at 80% power)
3. Added permutation test for pair count (p=0.609, descriptive only)
4. Added threshold sensitivity analysis
5. Predictive validity explicitly named as "the key open question"
6. Range-restriction paragraph with SD_outcome=0.052 vs SD_WMF-AM=0.228
7. Expanded construct map (6 columns: manipulated factor, observable, rivals, pattern, falsified if)
8. Added D'Amour et al. 2022 and Geirhos et al. 2020 references

### Loop 9 Round 2 Review (GPT-5.4 xhigh)

**Score: 7/10 (YES — ready for submission)**

Strengths:
- Central claim now well calibrated
- Theoretical rigor substantially better
- Pilot solid for position paper
- Honest about limitations (predictive validity deferred)
- Range restriction handled responsibly
- Taxonomy separation improved

Remaining weaknesses (no CRITICALs):
- MAJOR: Predictive validity still missing (acknowledged as future work)
- MAJOR: Empirical case local to narrow high-performing regime
- MINOR: Acronym density
- MINOR: Notation not fully stabilized
- MINOR: CoT interpretation should stay behavioral

### PDFs
- `cef_paper_v7_loop9_round0.pdf` — With 3 new experiments integrated
- `cef_paper_v7_loop9_round1.pdf` — After Round 1 fixes
- `cef_paper_v7.pdf` — Current (= loop9_round1)

---

## Loop 8 — N=15 Integration + Yoked Control (Mar 17, 2026)

**Major change**: Integrated N-expansion results (8 new Ollama models) and yoked cancellation control into paper. N=7→N=15.

### Loop 8 Round 0 (Pre-review edits)
- Abstract: N=7→N=15, τ=0.206→0.178, updated pair counts
- Section 4: Added 8 expansion models to pilot table with Yoked and WMF-Yoked columns
- Replaced inert control section with yoked cancellation control section
- Updated all statistics: τ(WMF-AM, Outcome)=0.178 (N=15), τ(WMF-Yoked, Outcome)=0.408 (p=0.047)
- Updated validity table: added WMF-Yoked vs Outcome row
- Updated limitations: N=7 objection partially addressed, new limitations (all open-weight, single-seed expansion)

### Loop 8 Round 1 Review (GPT-5.4 xhigh)

**Score: 5/10 (Almost)**

Strengths:
- Yoked cancellation control is strongest methodological element
- Better claim-evidence separation than earlier drafts
- Multi-seed and cross-template checks valuable
- Protocols as community resource

Weaknesses:
- CRITICAL: Dissociation claim overstated (tau=0.178 is non-significant, underpowered)
- CRITICAL: 20-item completion metric too coarse (resolution=0.05, threshold=noise)
- CRITICAL: K-depth inconsistency (WMF-AM {3,5,7} vs yoked {2,4,6,8,12} not clarified)
- MAJOR: Framework scope ahead of evidence (11 sub-dims, only 1 studied)
- MAJOR: Construct validity — many alternative explanations open
- MAJOR: AGENT-PQ independence concern (LLM judge)
- MAJOR: Not self-contained enough

Fixes implemented:
1. Fixed K-depth inconsistency in Section 3 and methods table
2. Reframed dissociation as "consistent with independence" not "proved"
3. Added explicit caveat about 20-item coarseness, future work calls for >=100 items
4. Removed r>=0.9 collapse criterion, softened to "working taxonomy"
5. Added "Alternative explanations" paragraph with explicit falsification criteria
6. Strengthened AGENT-PQ caveat as "necessary but not sufficient"
7. Enhanced methods table with Depths column and deployment details
8. Added multiplicity note for p=0.047 result
9. Added model-control details (chat templates, fp16, greedy decoding)
10. Shortened abstract to fix 27.8pt overfull

### Loop 8 Round 2 Review (GPT-5.4 xhigh)

**Score: 6/10 (Almost)**

Strengths:
- Claims-evidence alignment substantially improved
- Yoked control remains strongest contribution
- Methods table now auditable
- Alternative explanations + falsification improve rigor
- Reads appropriately as position/protocol paper

Weaknesses (no CRITICALs):
- MAJOR: Empirical base still thin (1 probe studied, 3 preliminary, rest protocols)
- MAJOR: Chat template heterogeneity confound
- MAJOR: Theory should distinguish independence vs incremental diagnostic value
- MAJOR: Completion baseline still weak (20 items)
- MAJOR: AGENT-PQ still hard to interpret
- MINOR: Uncertainty reporting light (point estimates + p-values)
- MINOR: Acronym load high
- MINOR: Exploratory/confirmatory boundary could be sharper

Remaining actionable items for future loops:
1. Template harmonization sensitivity check
2. Expand completion battery
3. Revise validity section theory (independence vs incremental value)
4. Human annotation for AGENT-PQ subset

### PDFs
- `cef_paper_v7_loop8_round0.pdf` — With N=15 data integrated (pre-review)
- `cef_paper_v7_loop8_round1.pdf` — After Round 1 fixes
- `cef_paper_v7.pdf` — Current (= loop8_round1)
