# Academic writing for the exam report

## Contents

1. The report is a short evidence story
2. Section blueprint (NeurIPS-style, adapted to an exam analysis)
3. Evidence binding: numbers, claims and citations
4. Calibrated language for this kind of analysis
5. Sentence- and word-level rules
6. Figures, tables and captions in the text
7. Pre-build self-review

Write in English, for a marker who knows statistics and machine learning and has ten minutes. The report must stand without the code. Every sentence either states what was done, states what was found (with its number), or bounds what the finding means.

## 1. The report is a short evidence story

State the one thing the reader should take away before writing anything else, in one sentence, for example: "A leakage-controlled linear pipeline classifies the five tumour types with no out-of-fold errors, and the class signal is concentrated in a small set of features." Every section exists to support, qualify or contextualise that sentence. If a result does not serve it, it goes to the appendix or is cut.

Spend roughly equal effort on (1) the abstract, (2) the introduction, (3) the figures and their captions, and (4) everything else. Markers read in that order and often stop early.

## 2. Section blueprint

Use this order unless the exam brief prescribes another. Headings are short noun phrases in sentence case.

**Abstract (one paragraph, 5 sentences).** (1) What was done and found; (2) why the task is hard (here: n ≪ p, class imbalance, leakage risk); (3) how, with the key method words a reader would search for; (4) the evidence (validation design, n, metric); (5) the most important number, stated with its evaluation kind. Never open with a sentence that could start any ML paper ("Machine learning has revolutionised…").

**1 Introduction.** The problem in two or three sentences; what makes it difficult; the approach and the one design decision that matters (e.g. all supervised steps refitted inside each fold); then 2–4 contribution bullets, each specific and checkable ("We compare three pipelines on identical frozen folds and select …", not "We conduct extensive experiments"). Optionally one roadmap sentence.

**2 Data.** Unit of observation, sample size, predictors, target classes and their counts, missing values versus true zeros, how labels were joined (by ID), excluded columns and why. Use the data_summary table rather than prose lists of numbers.

**3 Methods.** Enough to reproduce: the validation protocol first (split type, folds, seed, what was fitted inside folds), then each candidate pipeline as one sentence or a short list, then the selection rule and tie-break, then the exploratory analyses (tests, multiplicity correction, effect size, clustering pipeline) with their status labelled *exploratory*. Give formulas only where they remove ambiguity (η², BH).

**4 Results.** Follow the same order as the methods. Each subsection opens with the finding, then points to the figure or table, then gives the numbers that matter. Report the negative and the boring parts too: baseline, ties, which classes are confused, where unsupervised structure disagrees with labels.

**5 Discussion and limitations.** Answer the question at the level the evidence supports. Name concrete limitations with their likely consequence (selection CV is optimistic for the chosen model; no independent test set; ANOVA screen on all data is not predictive validation; anonymous features cannot be interpreted biologically; k fixed in advance). No generic "future work" padding.

**6 Conclusion.** Two to four sentences: the takeaway, the key number with its evaluation kind, the most important caveat.

**References.** Only works you actually relied on and can verify (software, statistical methods, the data source if the brief names it). No more than needed.

**Appendix.** Extra figures (e.g. the feature heatmap), extended tables, environment versions. The main text must stand without it.

## 3. Evidence binding

- Never type a result number. Use `{{fact}}` placeholders; list them with `build_reports.py --source report.md --list-facts`. If a needed number has no fact, compute it in the run outputs first, then add it, rather than writing it by hand.
- Name every number's evaluation kind at first use: *selection CV mean*, *pooled out-of-fold*, *exploratory all-data*, *public leaderboard*. Never call selection CV "test accuracy".
- Keep denominators visible: "801 of 801 out-of-fold predictions" is better than "100% accuracy".
- Ties and near-ties are reported as such, together with the declared tie-break.
- Citations: cite only entries you can verify (author, title, venue, year). Never invent a DOI, page range or author list. If unsure of a detail, omit the detail, not the honesty: a shorter correct reference beats a complete fabricated one. Do not cite to decorate; cite the software and methods you used.
- Do not claim novelty, biological meaning, clinical utility or causality. The data and design do not support them.

## 4. Calibrated language for this kind of analysis

| Say | Avoid |
|---|---|
| "achieves a selection-CV accuracy of 1.000 (3 folds, n = 801)" | "achieves perfect accuracy", "solves the problem" |
| "the classes are linearly separable in this sample under this protocol" | "the model will generalise perfectly" |
| "differs across classes (one-way ANOVA, BH q < 0.05)" | "is a biomarker for", "drives", "regulates" |
| "is associated with" | "causes", "determines" |
| "label-free clustering largely recovers the classes (ARI = 0.82)" | "the data naturally form five tumour types" |
| "we found no evidence of …" | "there is no …" (absence of evidence) |
| "feature gene_9175" | inventing a gene symbol for an anonymous column |

Hedge only where uncertainty is real, and then say exactly where it lies ("with three folds, the fold SD is a coarse stability indicator, not a confidence interval"). Do not hedge everything; do not overclaim anything.

Tense: present tense for what the report shows and for general facts ("Figure 2 shows", "the pipeline refits the selector in each fold"); past tense for what was done ("we standardised", "we compared"). Use "we" for the analyst's actions, active voice by default.

## 5. Sentence- and word-level rules

From Gopen and Swan's reader-expectation principles, as summarised in the ML-writing guides:

- Keep subject and verb close; put old information first and new, important information in the stress position at the end of the sentence.
- One paragraph, one point; the first sentence states it.
- Put the action in the verb ("we compared", not "we performed a comparison").
- Give context before the new thing ("To avoid leakage, we refit the selector inside each fold.").

Word choice:

- Prefer the specific word: "accuracy", "macro-F1", "η²" rather than "performance" or "effect"; "3-fold stratified" rather than "cross-validation" alone.
- One term per concept throughout: pick *sample* (not sample/patient/instance/case), *feature* (not gene/variable/predictor interchangeably, unless defined), *pipeline* (not model/method/approach interchangeably).
- Delete filler: very, really, quite, basically, essentially, actually, clearly, obviously, "it is worth noting that", "it should be noted that", "in order to" (→ "to").
- Avoid inflated or formulaic vocabulary unless it is the precise term: delve, pivotal, crucial, robust (outside its statistical meaning), comprehensive, leverage, showcase, underscore, landscape, realm, tapestry, intricate, multifaceted, groundbreaking, cutting-edge, paradigm.
- At most one or two em dashes in the whole report; use commas, parentheses or a new sentence. Semicolons sparingly.
- Vary sentence length naturally; do not force every argument into three points.
- Define every abbreviation at first use (OOF, BH, PCA, ARI, NMI) and use it consistently afterwards.
- Numbers: thousands separators (20,531), consistent decimals (three for scores, two for η² and ARI), units or scale stated once.

## 6. Figures, tables and captions in the text

- Every figure and table is referenced in the text before or where it appears (`@fig:models`, `@tab:cv_results`), and the text says what to look at ("Figure 1 shows that the two leading pipelines tie at …").
- Captions are generated from the run and already state the method and the limitation; the prose should interpret, not repeat the caption.
- Prefer one strong figure over three weak ones. In the main text, usually: model comparison, confusion matrix, one association figure and one structure figure. Heatmaps and extended distributions go to the appendix.
- Tables carry exact values (CV results, per-class recall, top features); figures carry patterns.

## 7. Pre-build self-review

Before running build_reports.py, reread the source against this list:

- [ ] The takeaway sentence is recognisable in the abstract, the introduction and the conclusion.
- [ ] Every number is a `{{fact}}` or comes from a generated table or caption.
- [ ] Each result names its evaluation kind; no "test accuracy" without an independent test.
- [ ] Exploratory analyses are labelled exploratory and kept separate from predictive validation.
- [ ] Limitations are concrete and each has a stated consequence.
- [ ] Contribution bullets are specific and checkable.
- [ ] Terminology is consistent; abbreviations are defined once.
- [ ] No filler or inflated words; at most two em dashes.
- [ ] Every citation is real and cited; every figure and table is referenced in the text.
- [ ] Nothing claims biological, clinical or causal meaning for anonymous features.
- [ ] The report reads as a paper: no code, file paths or tool names in the PDF prose (software is cited, not narrated).
