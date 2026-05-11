# Evaluation Metrics Summary

This document summarizes what the evaluation pipeline measures, how each metric is computed at a high level, and what each metric tells us about the generated Wikipedia-style island articles.

The main evaluation script is:

```bash
00_evaluation/full_evaluation.py
```

The script evaluates each generated article across five main metric groups:

1. **Writing quality**
2. **Informativeness**
3. **Concept-based evaluation**
4. **Citation and verifiability**
5. **CSCS**

Each row in the output CSV corresponds to one generated article for one island and one method.

---

## 1. What We Are Evaluating

The project generates Wikipedia-style articles about islands using different generation methods.

The evaluation asks:

* Is the article well written?
* Does it overlap with the reference article at the lexical level?
* Does it cover the important concepts from the reference article?
* Are the concepts accurate, relevant, and well organized?
* Are factual claims supported by citations?
* For Method 3, did the final article preserve the dependency structure produced by Agent 2?

The evaluation focuses on **article quality** and **Method 3 graph consistency**.

---

## 2. Evaluation Overview

| Metric group                  | What it evaluates                                                                          | Main output columns                                                                                                          |
| ----------------------------- | ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| Writing quality               | Fluency, structure, and organization of the generated article                              | `fluency_score`, `structure_score`, `organization_score`, `writing_score`                                                    |
| Informativeness       | Text overlap with the human Wikipedia reference                                            | `rouge_l`, `meteor`                                                                                                          |
| Concept-based informativeness | Whether important reference concepts are covered accurately and relevantly                 | `concept_coverage_score`, `concept_accuracy_score`, `concept_relevance_score`, `concept_organization_score`, `concept_score` |
| Citation/verifiability        | Whether claims are cited and whether citations support the claims                          | `citation_recall`, `citation_precision`, `citation_link_precision`, `citation_rate`                                          |
| CSCS                          | Whether Method 3 preserves dependency information across sections                          | `cscs`                                                                                                                       |

---

# 3. Writing Quality Metrics

Writing quality is evaluated by **Agent 4**, which acts as an LLM-based evaluator.

The evaluator reads the full generated article and assigns scores for writing quality.

## 3.1 `fluency_score`

**What it evaluates:**

Whether the article is grammatically fluent, readable, and natural.

A high fluency score means the generated text reads smoothly. A low fluency score indicates awkward phrasing, grammar issues, repetitive wording, or difficult-to-read sentences.

## 3.2 `structure_score`

**What it evaluates:**

Whether the article has an appropriate Wikipedia-style structure.

This includes whether the article has sensible sections, avoids malformed sections, and follows a coherent article format.

## 3.3 `organization_score`

**What it evaluates:**

Whether the information is logically ordered within and across sections.

A high organization score means the article develops ideas clearly. A low score suggests that facts may appear in an unnatural order, transitions may be weak, or related ideas may be scattered.

## 3.4 `writing_score`

**What it evaluates:**

The overall writing quality of the article.

This is a combined judgment of fluency, structure, and organization.

## 3.5 `writing_rationale`

**What it provides:**

A short explanation from Agent 4 explaining why the writing scores were assigned.

This is useful for qualitative interpretation of the writing metrics.

---

# 4. Informativeness Metrics

Lexical informativeness compares the generated article against a human-written Wikipedia reference article.

These metrics are computed in:

```bash
00_evaluation/metrics_informativeness.py
```

Before computing these scores, citation markers such as `[1]`, `[2]`, or `[1-3]` are removed so that citation formatting does not affect lexical overlap.

The scores are reported on a **0–100 scale**.

---

## 4.1 `rouge_l`

**What it evaluates:**

How much the generated article overlaps with the reference article in terms of longest shared token sequences.

ROUGE-L is based on the **longest common subsequence** between the generated article and the reference article.

**What a high score means:**

The generated article uses similar content, wording, and structure to the reference article.

**What a low score may mean:**

The article may miss important information, use very different wording, or be organized very differently from the reference.

**Important limitation:**

ROUGE-L is lexical. It does not directly prove that the article is factually correct. A generated article can have low ROUGE-L because it paraphrases well, or high ROUGE-L while still containing unsupported claims.

---

## 4.2 `meteor`

**What it evaluates:**

How much the generated article overlaps with the reference article using unigram matching.

METEOR is more flexible than exact sequence matching and is intended to capture word-level similarity between the generated and reference articles.

**What a high score means:**

The generated article covers many of the same words or concepts as the reference article.

**What a low score may mean:**

The article may omit important reference content or use substantially different wording.

**Important limitation:**

Like ROUGE-L, METEOR is still primarily an overlap-based metric. It should be interpreted together with the concept-based and citation-based metrics.

---

# 5. Concept-Based Evaluation Metrics

Concept-based informativeness is evaluated by **Agent 4** using the generated article and the human Wikipedia reference article.

Unlike ROUGE-L and METEOR, which measure surface-level overlap, this metric group asks whether the generated article captures the important concepts from the reference article.

Output columns include:

```text
concept_coverage_score
concept_accuracy_score
concept_relevance_score
concept_organization_score
concept_score
missing_key_concepts
inaccurate_or_unsupported_concepts
concept_rationale
```

---

## 5.1 `concept_coverage_score`

**What it evaluates:**

Whether the generated article includes the important concepts from the reference article.

For an island article, this may include topics such as geography, history, ecology, geology, settlement, administration, conservation, or notable events.

**High score:**

The generated article covers most of the important reference concepts.

**Low score:**

The article misses important topics or sections.

This metric is especially useful for identifying **content omission** errors.

---

## 5.2 `concept_accuracy_score`

**What it evaluates:**

Whether the concepts included in the generated article are factually accurate and supported by the reference/context.

**High score:**

The article presents concepts accurately.

**Low score:**

The article contains inaccurate, misleading, or unsupported information.

This metric is especially useful for identifying **content substitution** or hallucination-like errors.

---

## 5.3 `concept_relevance_score`

**What it evaluates:**

Whether the included concepts are relevant to the article topic.

**High score:**

The article stays focused on the island and includes relevant information.

**Low score:**

The article includes irrelevant, off-topic, or unnecessary content.

This metric is especially useful for identifying **content addition** errors.

---

## 5.4 `concept_organization_score`

**What it evaluates:**

Whether concepts are organized in a coherent and article-appropriate way.

**High score:**

Related concepts are grouped together and presented in a logical order.

**Low score:**

Important concepts may be scattered, repeated, or placed in confusing sections.

---

## 5.5 `concept_score`

**What it evaluates:**

The overall concept-level quality of the article.

This summarizes concept coverage, accuracy, relevance, and organization.

---

## 5.6 `missing_key_concepts`

**What it records:**

Important concepts found in the reference article but missing from the generated article.

This field provides interpretable evidence for content omission.

Example interpretation:

```text
If missing_key_concepts contains “ecological succession”, the generated article likely failed to cover an important aspect of the island.
```

---

## 5.7 `inaccurate_or_unsupported_concepts`

**What it records:**

Concepts in the generated article that appear inaccurate, unsupported, or inconsistent with the reference.

This field provides interpretable evidence for factual or grounding errors.

---

# 6. Citation and Verifiability Metrics

Citation and verifiability metrics are computed in:

```bash
00_evaluation/metrics_verifiability.py
```

These metrics evaluate whether generated factual claims are cited and whether those citations actually support the claims.

The evaluator uses:

1. Generated article sections.
2. Citation markers in the generated text.
3. Citation URLs attached to each section.
4. Retrieved source passages from the RAG context file.
5. An NLI model to test whether the source passage supports the cited claim.

The default NLI model is:

```text
cross-encoder/nli-deberta-v3-base
```

The default entailment threshold is:

```text
0.5
```

---

## 6.1 Sentence and citation processing

The verifiability evaluator first splits the generated text into sentences.

It then detects citation markers such as:

```text
[1]
[1][2]
[1, 2]
[1-3]
```

For each cited sentence, it maps each citation number to the corresponding URL in that section's citation list.

The evaluator also splits long sentences into smaller claim-like units. This is important because a citation may support one claim in a sentence even if it does not support the entire sentence.

---

## 6.2 `citation_rate`

**Definition:**

```text
citation_rate = cited sentences / total sentences
```

**What it evaluates:**

How often generated sentences contain citations.

**High score:**

Most sentences include citation markers.

**Low score:**

Many sentences are uncited.

**Interpretation:**

This measures citation density, not whether the citations are correct.

---

## 6.3 `citation_precision`

**Definition:**

```text
citation_precision = supported cited sentences / cited sentences
```

**What it evaluates:**

Among sentences that have citations, how many are actually supported by at least one cited source.

**High score:**

Most cited sentences are supported by their citations.

**Low score:**

The article may attach citations to claims that the sources do not support.

This is one of the most important verifiability metrics because it measures whether citations are meaningful rather than merely present.

---

## 6.4 `citation_recall`

**Definition:**

```text
citation_recall = supported cited sentences / total sentences
```

**What it evaluates:**

How much of the total article is both cited and supported.

**High score:**

A large portion of the article consists of supported, cited sentences.

**Low score:**

Many sentences are either uncited or not supported by the cited evidence.

---

## 6.5 `citation_link_precision`

**Definition:**

```text
citation_link_precision = supported citation links / total citation links
```

**What it evaluates:**

Whether individual citation links point to evidence that supports the cited claim.

**High score:**

Most citation links are useful and evidence-bearing.

**Low score:**

Many citation links are irrelevant, weak, missing from context, or unsupported.

This metric is stricter than sentence-level citation precision because it evaluates each citation link separately.

---

## 6.6 Diagnostic citation counts

The verifiability script also records:

```text
num_sentences
num_cited_sentences
num_citation_links
num_supported_citation_links
```

These counts help interpret the citation metrics.

For example, a high `citation_precision` with very few cited sentences may still indicate poor citation coverage.

---

# 7. CSCS: Cross-Sectional Consistency Score

CSCS is computed in:

```bash
00_evaluation/metrics_cscs.py
```

CSCS stands for **Cross-Sectional Consistency Score**.

It is only computed for:

```text
method3
```

because only Method 3 uses the Agent 2 dependency graph during generation.

For `method0`, `method1`, and `method2`, CSCS is marked as:

```text
not_applicable
```

---

## 7.1 What CSCS evaluates

Agent 2 creates a dependency graph between article sections.

For example, a downstream section may depend on information from an upstream section:

```text
child section -> parent section
```

CSCS checks whether information from the parent section is preserved or reflected in the child section.

This evaluates whether Method 3 actually uses the section dependency structure in a coherent way.

---

## 7.2 How CSCS is computed

For each dependency edge in the Agent 2 plan:

1. Find the generated text for the parent section.
2. Extract short fact-like units from the parent section.
3. Find the generated text for the child section.
4. Check whether the child section reflects each parent fact.
5. Use NLI entailment first.
6. If NLI does not mark the fact as entailed, use semantic similarity as a soft fallback.
7. Average the scores across checked facts.

---

## 7.3 `cscs`

**Score range:**

```text
0.0 to 1.0
```

**High score:**

Downstream sections preserve or reflect relevant upstream information.

**Low score:**

The generated article does not preserve dependency information across sections.

---

## 7.4 CSCS diagnostic columns

```text
cscs_status
cscs_edges
cscs_checked_edges
cscs_checked_facts
cscs_error
```

Interpretation:

* `cscs_edges`: number of dependency edges in the Agent 2 plan.
* `cscs_checked_edges`: number of edges considered by the evaluator.
* `cscs_checked_facts`: number of parent facts checked against child sections.
* `cscs_error`: explanation if CSCS could not be computed.


---

# 8. How to Interpret the Metrics Together

No single metric fully captures article quality. The evaluation is designed so that different metric groups explain different failure modes.

## Example interpretations

### High writing score but low citation precision

The article reads well, but the citations may not actually support the claims.

### High ROUGE-L but low concept accuracy

The article overlaps with the reference, but some concepts may still be wrong or unsupported.


### High concept coverage but low organization score

The article includes important information, but presents it in a confusing order.


### Low CSCS for Method 3

The generated article did not preserve dependency information between related sections.

---

# 9. Relationship to Error Analysis

The output of `full_evaluation.py` is used as the input to:

```bash
00_evaluation/error_analysis.py
```

---

# 10. Summary of What Each Metric Group Contributes

| Metric group             | Main purpose                                  | Best used for                                               |
| ------------------------ | --------------------------------------------- | ----------------------------------------------------------- |
| Writing quality          | Measures readability and article form         | Fluency, structure, organization                            |
| ROUGE-L / METEOR         | Measures overlap with reference article       | Surface-level informativeness                               |
| Concept-based evaluation | Measures semantic coverage and accuracy       | Missing concepts, unsupported concepts, relevance           |
| Citation/verifiability   | Measures citation support and coverage        | Grounding and factual support                               |
| CSCS                     | Measures dependency preservation for Method 3 | Cross-section consistency                                   |
| 

