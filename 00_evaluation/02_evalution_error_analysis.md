# Error Analysis Results

This error analysis summarizes the main article-quality failure modes across the evaluated outputs. Artifact and pipeline diagnostics are excluded, so the analysis focuses only on writing quality, content quality, citation/verifiability, and graph coherence.

The analysis covers **140 total outputs**.

## Reported fields

| Field | Meaning |
|---|---|
| `affected_outputs` | Number of generated outputs with at least one error of this class. |
| `percentage_of_outputs` | `affected_outputs / total_outputs_in_group`. |
| `high_severity_affected_outputs` | Number of outputs with at least one high-severity error of this class. |
| `medium_severity_affected_outputs` | Number of outputs with at least one medium-severity error of this class. |
| `total_error_instances` | Total number of error flags. One output can contain multiple error instances. |

---

# Error classification criteria

| Error source | Medium severity | High severity | Notes |
|---|---:|---:|---|
| Writing scores | score < 3.0 | score < 2.5 | Applies to fluency, structure, organization, and overall writing score. |
| Concept coverage | score < 3.0 | score < 2.5 | Measures missing reference-relevant concepts. |
| Concept accuracy | score < 3.5 | score < 3.0 | Uses a stricter cutoff because unsupported or inaccurate concepts directly affect factuality. |
| Concept relevance | score < 3.0 | score < 2.5 | Measures irrelevant or off-topic content. |
| Concept organization | score < 3.0 | score < 2.5 | Measures whether concepts are organized coherently. |
| Missing key concepts | 1 missing concept | 2+ missing concepts | Based on categorical evaluator output. |
| Inaccurate / unsupported concepts | any listed issue | any listed issue | Treated as serious because unsupported concepts directly affect factuality. |
| Citation rate | < 0.80 | < 0.50 | Share of factual sentences with citations. |
| Citation recall | < 0.20 | < 0.10 | Share of all sentences that are both cited and supported. |
| Citation precision | < 0.25 | < 0.10 | Share of cited sentences that are supported. |
| Citation link precision | < 0.10 | < 0.05 | Share of citation links that point to supporting evidence. |
| CSCS | < 0.50 | < 0.40 | Cross-section consistency score for dependency-linked sections. |
| ROUGE-L / METEOR | bottom 25% | bottom 10% | Distribution-based because these metrics do not have universal absolute cutoffs. |

The main setting uses both percentile-based thresholds and metric-specific absolute floors. This avoids using a single universal cutoff, such as 0.50, for every metric.

---

# Overall error frequency

| Error family | Error class | Affected outputs | % outputs | High-severity outputs | High % | Medium outputs | Medium % |
|---|---|---:|---:|---:|---:|---:|---:|
| Citation and verifiability | Citation support failure | 104 / 140 | 74.3% | 75 | 53.6% | 41 | 29.3% |
| Citation and verifiability | Citation coverage failure | 86 / 140 | 61.4% | 58 | 41.4% | 39 | 27.9% |
| Content quality | Content substitution | 77 / 140 | 55.0% | 6 | 4.3% | 71 | 50.7% |
| Planning and graph | Graph coherence failure | 44 / 140 | 31.4% | 12 | 8.6% | 32 | 22.9% |
| Content quality | Content omission | 43 / 140 | 30.7% | 19 | 13.6% | 27 | 19.3% |
| Writing quality | Writing error | 2 / 140 | 1.4% | 2 | 1.4% | 1 | 0.7% |
| Content quality | Content quality error | 2 / 140 | 1.4% | 0 | 0.0% | 2 | 1.4% |
| Content quality | Content organization error | 1 / 140 | 0.7% | 1 | 0.7% | 0 | 0.0% |

## Overall interpretation

The dominant errors are citation-related. Citation support failure affects **74.3%** of outputs, and citation coverage failure affects **61.4%**. This means many generated articles either do not cite enough factual claims or cite evidence that does not clearly support the sentence.

Content quality errors are present but less severe. Content substitution affects **55.0%** of outputs, but only **4.3%** are high-severity. This suggests that most concept accuracy issues are moderate rather than severe.

Writing quality is not a major failure mode. Only **1.4%** of outputs have writing errors, indicating that most articles are readable even when grounding or citation quality is weaker.

---

# Error frequency by method

| Method | Error class | Affected outputs | % outputs | High-severity outputs | High % |
|---|---|---:|---:|---:|---:|
| method0 | Citation coverage failure | 10 / 10 | 100.0% | 10 | 100.0% |
| method0 | Citation support failure | 10 / 10 | 100.0% | 10 | 100.0% |
| method0 | Content omission | 2 / 10 | 20.0% | 1 | 10.0% |
| method0 | Content substitution | 1 / 10 | 10.0% | 1 | 10.0% |
| method1 | Citation coverage failure | 6 / 10 | 60.0% | 2 | 20.0% |
| method1 | Citation support failure | 7 / 10 | 70.0% | 3 | 30.0% |
| method1 | Content omission | 7 / 10 | 70.0% | 4 | 40.0% |
| method1 | Content substitution | 8 / 10 | 80.0% | 0 | 0.0% |
| method2 | Citation coverage failure | 6 / 10 | 60.0% | 3 | 30.0% |
| method2 | Citation support failure | 7 / 10 | 70.0% | 5 | 50.0% |
| method2 | Content omission | 3 / 10 | 30.0% | 2 | 20.0% |
| method2 | Content substitution | 7 / 10 | 70.0% | 1 | 10.0% |
| method3 | Citation coverage failure | 64 / 110 | 58.2% | 43 | 39.1% |
| method3 | Citation support failure | 80 / 110 | 72.7% | 57 | 51.8% |
| method3 | Content omission | 31 / 110 | 28.2% | 12 | 10.9% |
| method3 | Content substitution | 61 / 110 | 55.5% | 4 | 3.6% |
| method3 | Graph coherence failure | 44 / 110 | 40.0% | 12 | 10.9% |
| method3 | Writing error | 2 / 110 | 1.8% | 2 | 1.8% |

## Method-level interpretation

**method0** has the weakest citation performance, with both citation coverage and citation support failures affecting all outputs. This is expected because it is the least citation-aware baseline.

**method1** improves citation coverage compared with method0, but content quality is weaker. Content omission affects **70.0%** of outputs, and content substitution affects **80.0%**.

**method2** reduces content omission from **70.0%** in method1 to **30.0%**, but citation support remains weak, with **70.0%** affected and **50.0%** high-severity.

**method3** gives the best overall balance. It keeps content omission relatively low at **28.2%**, and severe content substitution is rare at **3.6%**. The main remaining weakness is citation support, which affects **72.7%** of method3 outputs.

---

# Top concrete error subclasses

| Error subclass | Affected outputs | % outputs | High-severity outputs | High % |
|---|---:|---:|---:|---:|
| Low citation link precision | 97 / 140 | 69.3% | 73 | 52.1% |
| Low citation precision | 82 / 140 | 58.6% | 54 | 38.6% |
| Low citation recall | 77 / 140 | 55.0% | 57 | 40.7% |
| Low concept accuracy | 77 / 140 | 55.0% | 6 | 4.3% |
| Low CSCS | 44 / 140 | 31.4% | 12 | 8.6% |
| Low citation rate | 39 / 140 | 27.9% | 14 | 10.0% |
| Low METEOR | 35 / 140 | 25.0% | 14 | 10.0% |
| Low ROUGE-L | 35 / 140 | 25.0% | 14 | 10.0% |
| Low concept coverage | 4 / 140 | 2.9% | 4 | 2.9% |
| Low overall concept quality | 2 / 140 | 1.4% | 0 | 0.0% |
| Low overall writing quality | 2 / 140 | 1.4% | 1 | 0.7% |
| Poor organization | 2 / 140 | 1.4% | 2 | 1.4% |
| Poor concept organization | 1 / 140 | 0.7% | 1 | 0.7% |
| Poor article structure | 1 / 140 | 0.7% | 1 | 0.7% |

## Subclass interpretation

The most frequent concrete errors are citation-related: low citation link precision, low citation precision, and low citation recall. This shows that the main weakness is not prose quality, but grounding and evidence use.

Low concept accuracy is also common, but mostly medium-severity. Severe concept accuracy failures are rare.

Low CSCS affects **31.4%** of outputs, meaning that dependency-linked sections sometimes do not preserve or reflect upstream information consistently.

---

# Method3 ablation results

Each method3 ablation variant contains **10 outputs**.

| Variant | Citation coverage affected | Citation support affected | Content omission affected | Content substitution affected | Graph coherence affected |
|---|---:|---:|---:|---:|---:|
| default method3 | 50.0% | 80.0% | 30.0% | 40.0% | 40.0% |
| agent2 deepseek v3.2 | 60.0% | 90.0% | 40.0% | 50.0% | 30.0% |
| agent2 gemini 3 flash preview | 60.0% | 70.0% | 30.0% | 50.0% | 50.0% |
| agent2 qwen3.6 plus | 50.0% | 80.0% | 20.0% | 50.0% | 50.0% |
| agent3 claude opus 4.7 | 60.0% | 70.0% | 20.0% | 80.0% | 30.0% |
| agent3 gemini 3.1 pro preview | 70.0% | 80.0% | 30.0% | 80.0% | 40.0% |
| agent3 gpt5.4 | 70.0% | 90.0% | 30.0% | 60.0% | 30.0% |
| rerank scope global | 60.0% | 80.0% | 40.0% | 50.0% | 50.0% |
| reranker cross encoder + MMR | 40.0% | 40.0% | 20.0% | 50.0% | 30.0% |
| threshold 0.3 | 60.0% | 70.0% | 30.0% | 50.0% | 60.0% |
| threshold 0.6 | 60.0% | 50.0% | 20.0% | 50.0% | 30.0% |

## Ablation interpretation

The strongest ablation result is **cross encoder + MMR reranking**. It reduces citation coverage failure to **40.0%** and citation support failure to **40.0%**, compared with default method3 at **50.0%** and **80.0%**.

The **threshold 0.6** variant also improves citation support failure, reducing it to **50.0%**, while keeping content omission at **20.0%** and graph coherence failure at **30.0%**.

Agent2 model swaps are mixed. Qwen improves content omission to **20.0%**, but citation support failure remains **80.0%**. Gemini improves citation support to **70.0%**, but graph coherence worsens to **50.0%**.

Agent3 model swaps are also mixed. Claude reduces content omission to **20.0%** and graph coherence failure to **30.0%**, but content substitution rises to **80.0%**. GPT5.4 has lower graph coherence failure at **30.0%**, but citation failures remain high.

Overall, the ablation results suggest that retrieval and reranking changes are more beneficial than simply changing generation or planning models. The main bottleneck is still citation grounding rather than writing quality.

---

# Summary

The error analysis shows three main findings:

1. **Citation grounding is the main weakness.** Citation support and coverage failures dominate the error profile.
2. **Writing quality is generally strong.** Writing errors occur in only **1.4%** of outputs.
3. **Method3 is the strongest overall method, but reranking matters.** The cross encoder + MMR ablation gives the clearest improvement in citation-related errors.

The most important next improvement is better evidence selection and citation grounding, especially improving citation link precision and citation precision.