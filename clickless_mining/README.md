# ClickLess AI: data mining & evaluation data (Vikas)

Scope (proposal Table IV): **aisle-level Apriori, TF-IDF baseline, test prompt set and
ground-truth labels, Hit@3 evaluation.**

```
clickless_mining/
  tfidf.py          TF-IDF index (name, brand, category, ingredients, labels); Config A; Config B top-20 feed
  aisles.py         Instacart aisle -> catalog leaf map (33 aisles strong/partial + 4 optional; 101 unmapped)
  instacart.py      Instacart CSVs -> category transactions; split by user / by order
  apriori_rules.py  Apriori / FP-growth via mlxtend; single-consequent rules, lift > 1; held-out rule quality
  addons.py         RuleIndex (lift- or confidence-ranked add-on categories), recommend_addons, OfflineKG
  hit_eval.py       Hit@k / MRR, leave-one-out + first-item protocols, popularity + random baselines
clickless_eval/
  prompts.py        template rendering, annotation (acceptable sets), test-set validation
  metrics.py        CSS, TSR, IRA (+ infeasible P/R/F1, over-refusal), click model, Wilson / McNemar / bootstrap / Holm
scripts/
  tune_tfidf_dev.py     compare TF-IDF settings on the 20 DEV prompts only (settings then frozen)
  build_test_prompts.py gold specs -> paraphrase (llama3.1:8b, not the model under test) -> test_prompts.json
  run_config_a.py       Config A on the test set + TF-IDF Recall@20 (Config B ceiling)
  run_benchmark.py      Configs A, B, C side by side + paired tests + failure stages (+ add-ons if rules exist)
  run_apriori.py        Instacart -> rules (min_support picked on validation users) -> Hit@k on test users
```

## Setup

```bash
python3.13 -m venv .venv          # >= 3.11 (mlxtend 0.25 requires it)
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -m "not integration"
```

## Test prompts (data/evaluation/test_prompts.json)

80 single-item prompts: **70 feasible** (23 with 1 constraint, 24 with 2, 23 with 3+) and
**10 deliberately infeasible**. Every prompt has a gold constraint JSON and an acceptable set
that `acceptable_set` (same semantics as the Cypher template) computes from the catalog.
- Counting convention: each dietary tag, each excluded allergen and the price cap is one
  constraint; the category is always present; quantity is separate (20 prompts have qty > 1).
- Coverage: all 19 leaf categories plus 7 group prompts. 15 prompts are *tight* (1–2 acceptable
  products) and discriminate best between systems.
- Infeasible prompts stay in vocabulary, so they reach retrieval rather than the validator. Each
  has an `infeasible_reason`: tag×category, allergen×category, price below the category minimum,
  tag conjunction, tag×allergen, three-way, or group level.
- Wording: template → paraphrased by local **llama3.1:8b** (not qwen2.5:7b, the model under test)
  → reviewed by hand. **47 of 80 paraphrases changed a constraint and were corrected.** Typical
  errors: an invented price cap or label, a narrowed category (e.g. "potato chips"), a dropped
  constraint, or an allergen exclusion reworded as a "gluten-free" label. The original text and
  the reason for each correction are kept in `data/evaluation/test_prompt_paraphrases.json`.
- `template` holds the canonical wording, so a run on template text versus paraphrases measures
  sensitivity to wording.
- The 20 prompts in `constraint_prompts.json` shaped the extractor prompt and the TF-IDF settings,
  so they are the **development set**. The validator rejects any test prompt that duplicates them.
- TODO (team): a second annotator should check ≥20 prompts, including all 10 infeasible ones, and
  report Cohen's κ. Then add their name to `verified_by`. The current `verified_by` is
  `claude-review`, meaning an AI-assisted review that a human still needs to confirm.

## TF-IDF baseline

scikit-learn defaults: unigrams, raw tf, smoothed idf `ln((1+n)/(1+df))+1`, L2 rows. Cosine
similarity = `linear_kernel`. Ties are broken by barcode.
- Price, allergens and barcodes are not indexed.
- Config A always returns a product, so **IRA(A) = 0 by construction**.
- Config B receives `tfidf_candidates(query, index)` (the top 20 catalog dicts) and passes them
  unchanged to `select_from_candidates`.

| Config A (80 test prompts) | value |
|---|---|
| CSS | 0.766 |
| TSR | 0.529 (95% Wilson CI 0.41–0.64) |
| IRA | 0.000 (by construction) |
| TF-IDF Recall@20 (Config B ceiling) | 0.971 |
| MRR | 0.657 |

Violations on feasible prompts: allergen 15, tag 13, price 10, category 10.

## Instacart → Apriori → Hit@3

The data needs a Kaggle account with the competition rules accepted. The terms forbid
redistribution and the files are over 100 MB, so `data/instacart/` is gitignored. Commit only
`data/addon_rules.json` (derived, category-level).

**Copy used for our runs.** The Kaggle competition page was unreachable on 2026-11-22, so the
files came from the public Hugging Face copy `attik/Instacart-Market-Basket-Analysis`. It was
checked against the original release:
- aisles 134, departments 21, products 49,688;
- orders 3,421,083 (prior 3,214,874 / train 131,209 / test 75,000);
- order_products__prior 32,434,489 and order_products__train 1,384,617;
- original column headers;
- every mapped aisle id and name matches `aisles.csv` (`aisles.validate_against_aisles_csv`).

Cite it as "The Instacart Online Grocery Shopping Dataset 2017". The original terms still apply:
non-commercial use, and no redistribution.

```bash
.venv/bin/kaggle competitions download instacart-market-basket-analysis -p data/instacart
cd data/instacart && unzip -o instacart-market-basket-analysis.zip && for f in *.zip; do unzip -o "$f"; done; cd -
.venv/bin/python scripts/run_apriori.py                    # primary: split by user, seed 573
.venv/bin/python scripts/run_apriori.py --split order --rules-out /tmp/r.json --out data/evaluation/hit_at_3_order_split.json
```

Protocol:
1. 80/20 split **by user**.
2. The ranking (lift or confidence) and `min_support` are chosen *jointly* by Hit@3 on a
   validation 10% of the training users.
3. Rules are re-mined on the full 80%.
4. The test users are scored once, on a seeded sample of 100k queries per protocol.

Scoring:
- Two protocols: **leave-one-out** (hide one category, primary) and **first-item** (cart = the
  first category added).
- Recommenders: lift-ranked rules (with popularity back-fill, and rules only), confidence-ranked
  rules, **most popular** and random.
- Reported: Hit@1/3/5, MRR, rule coverage, exact McNemar with Holm correction, and the held-out
  support/confidence/lift of the top 20 rules.

### Results (data/evaluation/hit_at_3.json, run 2026-11-22)

**Coverage.** 34.4% of the 33.8M order lines map to a catalog leaf. 2.83M orders keep ≥1
category and 2.13M keep ≥2 (mean 3.07 categories per order; 202,561 users). The split gives
2.27M training orders and 567k test orders.

**Selection on validation.**
- Lift-ranked rules scored only 0.166–0.426, rising as min_support rose toward the point where
  rules vanish and popularity takes over.
- Confidence-ranked rules scored 0.432 at min_support 0.0005, against 0.426 for most-popular.
- Selected: **confidence ranking, min_support = 0.0005, max_len = 3**. That gives 3,166 rules
  (lift > 1, one consequent) from frequent itemsets per level |F1| = 19, |F2| = 171, |F3| = 968.

**Test, 100k queries per protocol, rule coverage 1.00:**

| Hit@k | Apriori (conf.) | Apriori (lift) | Most popular | Random |
|---|---|---|---|---|
| leave-one-out Hit@1 | **0.182** | 0.058 | 0.173 | 0.063 |
| leave-one-out Hit@3 | **0.432** (0.429–0.435) | 0.163 | 0.426 (0.423–0.429) | 0.188 |
| leave-one-out Hit@5 | **0.603** | 0.286 | 0.602 | 0.314 |
| first-item Hit@3 | 0.638 | 0.383 | 0.638 | 0.380 |

- On leave-one-out Hit@3, confidence-ranked rules beat most-popular (exact McNemar b = 2974,
  c = 2376, Holm p = 3e-16). The gain is small: **+0.6 points**.
- On the first-item protocol they tie (p = 0.23).
- **Lift-ranked rules (the proposal's wording) fall below random.** Lift rewards rare consequents
  and ignores how often the consequent is bought, the known prevalence bias. Report this
  honestly as a finding.
- The top rules hold up on held-out users. The best one is
  `crackers, pasta_sauce -> cheese [sup = 0.008, conf = 0.634, lift = 1.83]`, with held-out
  confidence 0.642 and lift 1.86.
- Robustness runs. The same pattern holds in both:

  | Run | LOO Hit@3 conf / pop / lift / random | First-item Hit@3 conf / pop |
  |---|---|---|
  | `hit_at_3_order_split.json` (random 20% of orders, the proposal's literal wording) | 0.431 / 0.426 / 0.166 / 0.190 | 0.637 / 0.638 |
  | `hit_at_3_optional_aisles.json` (4 weak aisles also mapped; min_support 0.0025 selected) | 0.441 / 0.433 / 0.164 / 0.190 | 0.647 / 0.649 |

## A/B/C benchmark (data/evaluation/benchmark_results.json, run 2026-11-22)

80 test prompts. B and C use qwen2.5:7b at temperature 0. The KG is `OfflineKG`, which has the
same semantics as the Cypher query.

| Config | CSS | TSR (95% CI) | IRA (95% CI) | infeasible F1 | over-refusal | median s/prompt |
|---|---|---|---|---|---|---|
| A TF-IDF | 0.766 | 0.529 (0.41–0.64) | 0.00 | 0.00 | 0.000 | <0.01 |
| B LLM over TF-IDF top-20 | 0.682 | 0.629 (0.51–0.73) | 1.00 (0.72–1.00) | 0.49 | 0.300 | 22.7 |
| C ClickLess (LLM + validator + KG) | **0.929** | **0.929** (0.84–0.97) | 1.00 (0.72–1.00) | **0.80** | 0.071 | 4.4 |

Paired tests on the 70 feasible prompts (exact McNemar on TSR, bootstrap on CSS, Holm across all):
- **C vs B:** TSR p_Holm = 0.0001; CSS +0.246 (0.129 to 0.364).
- **C vs A:** TSR p_Holm < 0.0001; CSS +0.162 (0.068 to 0.251).
- **B vs A:** TSR p_Holm = 0.11, not significant.

Failure stages:
- **C:** 5 failures, all at extraction, giving constraint-extraction exact match 0.90. Four of the
  five map "plant-based milk" to the leaf `milk` instead of `plant_milk`, so the extractor prompt
  needs a plant-milk rule. In the fifth, "no gluten" was extracted as the `gluten_free` label
  rather than an excluded allergen.
- **B:** 19 false "no product" answers, 5 violating picks (tag or price), and 2 cases with no
  acceptable product in the top 20.
- **A:** the main error types are allergen, tag, price and category violations, as expected for
  a lexical retriever.

Add-ons on Config C matches: 65 of 65 matched prompts got 3 add-ons, and **100% of the add-on
products satisfy the user's dietary tags and excluded allergens**.

### After error analysis (v2, `benchmark_results_v2_after_error_analysis.json`)

**Report v1 (frozen, above) as the primary result.** v2 changed two things in response to the
v1 failures, so its numbers are post-hoc:
- **Extractor:** a plant-milk rule (and an explicit rule that "lactose-free milk" is dairy
  `milk`). The dev set stays at 95% exact match.
- **Config B:** now returns a quantity, so its TSR checks quantity.

| Config | CSS | TSR (incl. quantity for B/C) | IRA | infeasible F1 | over-refusal |
|---|---|---|---|---|---|
| A | 0.766 | 0.529 | 0.00 | 0.00 | 0.000 |
| B | 0.772 | 0.671 | 0.60 | 0.43 | 0.171 |
| C | **0.986** | **0.986** | **1.00** | **0.95** | 0.014 |

- **C vs B:** TSR Holm p < 0.0001; CSS +0.214 (0.125 to 0.309).
- **C remaining failure:** 1 (extraction). Constraint-extraction exact match is 0.90; the misses
  are in fields that don't change the acceptable set.
- **Add-ons:** 69 of 69 matched prompts got 3 add-ons, and 100% are constraint-compliant.
- A first v2 attempt over-corrected: "lactose-free milk" became `plant_milk` and "gluten-free"
  became an allergen exclusion (C TSR 0.957, IRA 0.90). It was replaced, as documented here.

### v3: grounding guard, run on live Neo4j (`benchmark_results_v3_grounding_neo4j.json`)

Live UI tests showed qwen2.5:7b still inventing tags ("oat milk" → vegan, non_gmo,
gluten_free) and mixing up dairy and plant milk on new wordings. `clickless_llm/grounding.py`
is a deterministic post-extraction guard. It drops any dietary tag whose trigger words are not
in the request and picks milk or plant_milk from the request words. It never adds a constraint.
- Config C on **Neo4j** (Docker): CSS 0.986, TSR 0.986, IRA 1.00, infeasible F1 0.95.
- Extraction exact match is **0.94** (v2: 0.90; v1: 0.90). The dev set stays at 95%.

**Neo4j = OfflineKG.** `scripts/verify_kg_equivalence.py` shows the Cypher template and the
offline filter agree:
- 80 of 80 gold acceptable sets are identical;
- 160 of 160 Config C top-5 lists (from the cached extractions) are identical;
- Samuel's integration tests pass on the real catalog (4 of 4).

### Click reduction (`scripts/click_reduction.py`; `click_reduction_v1.json`, `click_reduction_v2.json`)

Manual flow over the 70 feasible prompts: 5.73 clicks on average. The agent needs 2 (one
submission and one approval; automated runs have no corrections).

| | solved feasible | mean manual clicks on solved | mean click reduction |
|---|---|---|---|
| B v1 / v2 | 44 / 47 | 4.75 / 4.72 | 57.0% / 56.7% |
| C v1 / v2 | 65 / 69 | 5.83 / 5.72 | **60.9% / 60.3%** |

## Add-ons in the pipeline

```python
from clickless_mining.apriori_rules import load_rules
from clickless_mining.addons import RuleIndex, recommend_addons
rules = load_rules("data/addon_rules.json")
index = RuleIndex(rules["rules"], rules["popularity"], rank_by=rules["meta"]["rank_by"])   # "confidence"
addons = recommend_addons([product["category"]], constraints, index, kg, k=3)
# each: {"category", "source": "rule"|"popularity", "rule": {antecedents, consequent, support, confidence, lift}, "product"}
```
`kg` is a `KGRetriever` (Neo4j) or `OfflineKG(catalog)`. Show the rule as the explanation:
`format_rule(addon["rule"])` gives `milk -> breakfast_cereal [sup = .., conf = .., lift = ..]`.

## Third-party code (for the report)

scikit-learn (TfidfVectorizer, linear_kernel), mlxtend (apriori, fpgrowth, association_rules),
pandas, numpy, the Kaggle CLI. Everything else in `clickless_mining/`, `clickless_eval/` and the
scripts listed above was written for this project.
