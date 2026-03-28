# Paper Improvement Log — CEF NeurIPS 2026

## Score Progression

| Round | Score | Verdict | Key Changes |
|-------|-------|---------|-------------|
| Round 0 (original) | 5/10 | Almost | Baseline: diffuse center, confirmatory/exploratory not separated, dense abstract |
| Round 1 | 6/10 | Almost | Abstract rewritten, study map table, variable naming, CEF compressed, subsection headers |
| Round 2 | 6/10 | Almost | ABS task table added, secondary stats compressed, Completion Fallacy softened |

## Round 1 Review & Fixes

<details>
<summary>GPT-5.4 xhigh Review (Round 1) — Thread: 019d1498-d10e-7fb1-b3e7-2cfef831c8b2</summary>

**Overall Score:** 5/10

**Summary:**
The paper has a potentially strong central story: a calibrated no-scratchpad cumulative arithmetic state-tracking probe appears predictive of downstream deterministic agent performance across 20 open-weight LLMs. The main writing problem is not lack of material, but lack of focus: the core contribution is buried under competing frames, too many acronyms, and too many secondary analyses in the main text.

**Weaknesses:**
- CRITICAL 1: Paper lacks a clean center of gravity. WMF-AM validation, "Completion Fallacy," and CEF taxonomy compete.
- CRITICAL 2: Confirmatory and exploratory results not separated sharply enough in presentation.
- CRITICAL 3: Core variables not named consistently.
- MAJOR 4: Abstract is too dense.
- MAJOR 5: Downstream agent battery under-described in main paper.
- MAJOR 6: Acronym density too high.
- MAJOR 7: Main results section reads like a compressed appendix.
- MAJOR 8: Table captions weaken professionalism and self-containedness.
- MINOR 9-12: Rhetorical overclaims, list-like structure, formal definition overstated, title too long.

**Verdict:** Almost

</details>

### Fixes Implemented (Round 1)
1. Abstract rewritten: 4-item structure (Question / Method / Main result / Scope); secondary stats deferred to last sentence
2. Variable definitions paragraph added at start of Section 4: WMF-AM Score, Outcome Correctness (OC), Agent Battery Score (ABS)
3. Study map table (tab:keyanalyses) caption updated with C/E definitions and variable names
4. CEF section restructured: bypass note, "CEF background (skippable)" label, MCC/EMC/CLA collapsed to one paragraph with appendix reference
5. Results section reorganized into explicit subsections: §4.1 Primary Confirmatory, §4.2 Robustness, §4.3 Exploratory, §4.4 Measurement Robustness
6. Exploratory observations paragraph compressed; complementary probes paragraph halved
7. tab:pilot renamed with descriptive caption identifying WMF-AM Score / OC / ABS with C/E labels
8. Predictor comparison table caption clarified: criterion=ABS, delta-tau bootstrap results explicit

---

## Round 2 Review & Fixes

<details>
<summary>GPT-5.4 xhigh Review (Round 2) — Thread: 019d1498-d10e-7fb1-b3e7-2cfef831c8b2</summary>

**Overall Score:** 6/10

**Summary:**
The manuscript is materially improved. The paper now has a clearer empirical spine, better confirmatory vs exploratory separation, and more disciplined naming, which removes the biggest Round 1 writing blockers. The remaining issues are mostly about self-containedness and compression rather than contribution confusion.

**Remaining Weaknesses:**
- MAJOR: ABS still under-described — needs a compact, concrete, self-contained presentation in main paper
- MAJOR: Too many secondary statistics still in main text
- MAJOR: "Completion Fallacy" framing still somewhat polemical in high-visibility locations
- MINOR: Acronym density still above ideal
- MINOR: Some presentation choices still appendix-like
- MINOR: Title still longer and denser than optimal

**Minimum Actionable Fixes:**
1. Add one compact ABS table (task name, capability, scoring rule) before primary results
2. Keep only primary result + one predictor table + one robustness summary; move exact secondary values to appendix
3. Soften "Completion Fallacy" framing in high-visibility locations; use "limits of completion-only evaluation" as default

**Verdict:** Almost

</details>

### Fixes Implemented (Round 2)
1. Added compact ABS task table (tab:battery): 10 tasks with category, capability, and deterministic scoring rule; leave-one-category-out results in caption
2. Compressed confound controls to single paragraph with appendix reference; moved leave-one-out, task-split, and secondary stats to one sentence each
3. Section 2 title changed from "The Completion Fallacy..." to "Limits of Completion-Only Evaluation..."
4. Introduction framing softened: "limits of completion-only evaluation" as primary label, "Completion Fallacy" relegated to parenthetical shorthand
5. Fixed Unicode τ character → LaTeX `$\tau$` at line 687 (missing character warning eliminated)
6. Fixed 10.78pt overfull hbox → 3.38pt via `\sloppy` in CEF background paragraph

---

## PDFs
- `cef_paper_v7_pil_round0_original.pdf` — Original paper (284.26 KiB)
- `cef_paper_v7_pil_round1.pdf` — After Round 1 fixes (284.27 KiB)
- `cef_paper_v7_pil_round2.pdf` — Final after Round 2 fixes (283.82 KiB)

## Final Format Check
- PDF size: 283.82 KiB
- Overfull hbox >10pt: 0 (down from 1)
- Remaining overfull: 3.38pt (within tolerance)
- Missing character warnings: 0 (Unicode τ fixed)
- Compile: Clean (0 errors)

## Remaining Issues (non-blocking; future work)
- Title remains verbose (not shortened; changing title would require re-syncing with overleaf zip)
- Acronym density (WMF-AM, CEF, MCC, EMC, CLA) still present but all used in main text
- Formal "statistical insufficiency" definition in Section 2 still appears authoritative; future revision could add explicit "exploratory" qualifier

---

# PIL Run 2 (Thread: 019d14c6-866e-79a0-b8a6-6faabc171177)

## Score Progression

| Round | Score | Verdict | Key Changes |
|-------|-------|---------|-------------|
| Round 0 (PIL2 baseline) | 7/10 | Almost | Starting point after PIL1 |
| Round 1 | 7/10 | Almost | (Review received; fixes pending) |
| Round 2 | **8/10** | **Ready** | Claim-hierarchy ¶, sentence-level labels, abstract compressed, limitations 4-bucket reorganization |

## Round 1 Review (PIL2)

**Score: 7/10 — Almost**

MAJOR issues:
1. Claim-hierarchy paragraph missing at end of introduction
2. Sentence-level inferential labels needed (not just subsection labels)
3. Abstract still too numerically dense
4. Limitations section: 10 items → needs 3-4 ranked buckets

MINOR: terminology load, table captions, Section 2 over-argued, title length

## Round 1 Fixes Implemented (PIL2)

1. Added `\paragraph{Claim hierarchy}` at end of introduction: sole empirical contribution = WMF-AM validation; all else motivates/stress-tests/hypothesis-generates; CEF is interpretive framework
2. Added "Robustness check ---" label to held-out model paragraph; "Exploratory:" prefix to log(params) partial τ in robustness section
3. Compressed abstract: removed τ=0.627 number, bundled 3 ablations into 1 sentence, removed K-range [0.050, 0.983], added explicit scope sentence about sole empirical contribution
4. Reorganized limitations into L1 (external validity), L2 (construct validity), L3 (sample size/dependence), L4 (comparator scope)
5. Fixed 3 new overfull hboxes (22pt, 89pt, 10pt) → only 3.38pt remains

## Round 2 Review (PIL2) — GPT-5.4 xhigh

<details>
<summary>Full raw review (click to expand)</summary>

**Score:** 8/10

**Summary**
These revisions address the main writing-level risks from the prior round. The paper now reads as having a clear empirical center of gravity, stronger inferential hygiene, and a much better-balanced abstract-to-limitations arc. On writing quality alone, I would no longer treat submission readiness as blocked.

**Strengths**
- The new claim-hierarchy paragraph is exactly the right editorial intervention: it sharply distinguishes the paper's single empirical contribution from framework, motivation, and exploratory material.
- Sentence-level inferential labels materially improve reviewer trust. They reduce the chance that exploratory or robustness results are read as confirmatory.
- The abstract is now better scoped and easier to parse; removing excess numeric detail was the right choice.
- The limitations section sounds more mature after reorganization into ranked buckets. It now communicates seriousness without feeling like an undifferentiated caveat dump.
- The paper's internal vocabulary appears more controlled: WMF-AM, WMF-AM Score, OC, and ABS now seem anchored to a cleaner empirical story.
- The confirmatory/exploratory distinction is now a genuine paper-wide writing feature, not just a section-label convention.
- The overall narrative arc is stronger: motivation -> probe -> validation -> robustness/ablations -> exploratory extensions -> limitations.

**Weaknesses**
- CRITICAL: None.
- MAJOR: None.
- MINOR:
  - The title is still long and cognitively dense. It signals rigor, but it may slow first-pass comprehension.
  - The abstract may still be close to the upper limit of conceptual density because it combines contribution, ablation summary, positioning against Minerva, and scope language in a short space.
  - Section 2 and the CEF material remain the places most likely to feel slightly over-proportioned relative to the core validation story; this is now manageable, but still worth watching in final polishing.
  - Table-caption self-containment is still something to verify carefully in the actual PDF, especially for the key analyses, predictor comparison, K-sweep, and validity-summary tables.
  - The reorganized limitations are much better, but the prose should still ensure that L1 and L3 clearly dominate reviewer takeaway, with L2 and L4 presented as secondary rather than co-equal threats.

**Verdict:** Ready

On the basis of the changes described, this is now at a submission-ready writing standard for a NeurIPS main-track paper. One last polish pass for title/caption economy and section-balance discipline, but submission is not blocked.

</details>

## Format Check (PIL2 Round 2)
- PDF size: 284 KiB
- Overfull hbox >10pt: 0
- Remaining overfull: 3.38pt (within tolerance)
- Compile: Clean (0 errors)

## Remaining Issues (MINOR; non-blocking)
- Title is verbose — consider shortening before submission
- Abstract slightly dense — could trim 1 sentence in final polish
- Section 2 / CEF material slightly over-proportioned vs core story
- L1/L3 limitations should be emphasized over L2/L4 in final read

---

# PIL Run 3 — Format Compliance (Thread: 019d1526-0372-7742-8c68-dfd705486c0c)

## Problem Identified
Main body was 15 pages — 6 pages over NeurIPS 2026 9-page limit.

## Score Progression

| Round | Score | Verdict | Key Changes |
|-------|-------|---------|-------------|
| Round 0 (PIL3 baseline) | 8/10 | Ready | Starting point: 15-page main body |
| Round 1 | 8/10 | **Ready** | Cut 6 pages: main body now exactly 9 pages |
| Round 2 | 8/10 | **Ready** | MINOR fixes: alt-positioning sentence, validity summary C/E labels, empty citation removed |

## Cuts Implemented

**Section 2**: Collapsed from 2 full pages to 1 paragraph + formal definitions. All F1/F2/F3 failure modes, empirical support, positioning, and Minerva comparison table moved to new Appendix `app:related_full`. Added 1 sentence pointing to Minerva as closest prior.

**Section 3**: Removed CEF background (skippable) paragraph, CEF design principles, tab:dims table, "Other CEF dimensions" paragraph, "Construct validity framework" paragraph — all moved to new Appendix `app:cef_background`. Kept only WMF-AM probe description + worked examples.

**Section 4**: Replaced §4.3 "Exploratory Analyses" (ABS task table, yoked cancellation detail) with 1 paragraph + predictor comparison table. Removed full validity summary table (moved to appendix); replaced with 4-sentence text summary.

**Section 5**: Removed "Outcomes vs. process" and "Cognitive paradigms" paragraphs. Compressed Minerva comparison from ~1 page (incl. K-sweep table) to 1 paragraph; K-sweep table moved to new Appendix `app:ksweep`. Compressed discussion ending to 1 combined paragraph.

## Format Check (PIL3 Round 2)
- **Main body pages: 9** (bibliography starts page 10, appendix pages 12-20)
- Total pages: 20
- PDF size: 221 KiB
- Overfull hbox >10pt: 0
- Remaining overfull: 1.30pt, 6.59pt (both below 10pt threshold)
- NeurIPS 2026 compliance: ✓

## Round 1 Review (PIL3) — GPT-5.4 xhigh

Score: **8/10 — Ready for NeurIPS 2026 submission**

Key findings: Cuts are broadly coherent. Preserved epistemic discipline (claim hierarchy, scope, limitations) matters more than keeping every exploratory table in-body. Two borderline risks (CEF under-justification, over-compressed validity narrative) not triggered. Remaining MINOR issues: Section 2 proximity to alternatives (fixed), appendix pointer specificity (already specific), validity summary claim-type labels (fixed), empty citation (fixed).

## Final State
- Paper is 8/10 Ready for NeurIPS 2026 main track
- Main body: 9 pages ✓
- All CRITICAL/MAJOR issues resolved
- arXiv-ready after final title polish (MINOR: title still verbose)
