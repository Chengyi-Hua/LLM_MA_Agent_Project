# Evaluation Metrics, Interpretation, and Method-Level Findings

## 1. Evaluation Setup

This evaluation compares four generation methods across 10 island cases.

| Method | Description |
|---|---|
| `method0` | Pure generation baseline without retrieval or citations |
| `method1` | Naive RAG |
| `method2` | Hierarchical RAG |
| `method3` | Agent 2 dependency planning + Agent 3 context-aware generation |

The reported values are descriptive averages over 10 islands:

`mean ± standard deviation (n=10)`

Because the sample size is small, the results should be interpreted as descriptive trends rather than statistically proven differences unless paired significance tests or bootstrap confidence intervals are added.

---

## 2. Metric Definitions and Interpretation

### 2.1 Writing Quality Metrics

The writing metrics are judged by Agent 4.

| Metric | Interpretation |
|---|---|
| `fluency_score` | How natural, readable, and grammatically fluent the article is |
| `structure_score` | Whether the article has an appropriate encyclopedic structure |
| `organization_score` | Whether ideas and sections are logically ordered |
| `writing_score` | Overall writing quality, combining fluency, structure, and organization |

These scores mainly measure surface-level article quality. They do not measure whether claims are supported by evidence. A high writing score means that the article reads well, but it does not necessarily mean that the article is factually grounded.

### 2.2 Lexical Informativeness Metrics

| Metric | Interpretation |
|---|---|
| `ROUGE-L` | Measures longest-subsequence lexical overlap with the reference article |
| `METEOR` | Measures unigram-level overlap with the reference article, with more flexibility than ROUGE |

These metrics estimate how much the generated article overlaps with the human-written reference article. They are useful for measuring reference similarity, but they are limited because two articles can express the same facts using different wording.

In the corrected evaluation, citation markers such as `[1]`, `[2]`, or `[1, 2]` are removed before computing ROUGE-L and METEOR. This avoids unfairly penalizing citation-heavy RAG outputs.

### 2.3 Concept-Based Metrics

The concept metrics are also judged by Agent 4 against the reference article.

| Metric | Interpretation |
|---|---|
| `concept_coverage_score` | Whether important reference concepts are included |
| `concept_accuracy_score` | Whether included concepts are factually correct |
| `concept_relevance_score` | Whether the generated content stays relevant to the island |
| `concept_organization_score` | Whether concepts are organized coherently |
| `concept_score` | Overall concept-level quality |

These metrics are important because lexical overlap alone may miss semantically correct paraphrases. A high concept score means the article covers relevant ideas well, but it does not automatically mean that every statement is citation-supported.

### 2.4 Citation and Verifiability Metrics

The citation metrics evaluate whether generated sentences are supported by retrieved evidence.

| Metric | Definition | Interpretation |
|---|---|---|
| `citation_rate` | cited sentences / total sentences | How often the article uses citations |
| `citation_precision` | supported cited sentences / cited sentences | Among cited sentences, how many are supported |
| `citation_recall` | supported cited sentences / total sentences | How much of the whole article is citation-supported |

In this evaluation, `citation_recall` is operationalized as supported sentence coverage. It is not classical information-retrieval recall. It does not measure all possible relevant evidence. Instead, it measures the proportion of generated sentences that are both cited and judged supported.

A high citation rate alone is not enough. A method can cite nearly every sentence but still have low citation precision if the citations do not actually support the claims.

### 2.5 Cross-Sectional Consistency Score, CSCS

CSCS is only applicable to `method3`.

`method3` uses the Agent 2 dependency graph during generation. Therefore, CSCS evaluates whether information from upstream sections is preserved or reflected in downstream sections.

| Metric | Interpretation |
|---|---|
| `cscs` | Average parent-child dependency reflection score across planned section dependencies |

CSCS ranges from 0 to 1.

| Score | Meaning |
|---|---|
| `0.0` | Downstream sections do not reflect upstream dependency information |
| `~0.5` | Moderate semantic reflection of upstream information |
| `1.0` | Strong entailment or preservation of upstream facts |

CSCS should not be compared across all methods because only `method3` uses the dependency graph. For `method0`, `method1`, and `method2`, CSCS is not applicable, not zero.

---

## 3. Results by Metric Family

### 3.1 Writing Quality

| Method | Writing Score |
|---|---:|
| `method0` | `4.0650 ± 0.3432` |
| `method1` | `3.8340 ± 0.2357` |
| `method3` | `3.4660 ± 0.1756` |
| `method2` | `3.4000 ± 0.3788` |

`method0` achieves the highest writing quality. It also has the highest fluency score:

`method0 fluency_score = 4.4000`

This suggests that pure generation produces the most polished prose. However, this does not mean `method0` is the best RAG method. It produces fluent text, but it does not provide citations or evidence grounding. Zhis only means that written text is good, nothing about the content and validity behind.

The main writing-quality finding is:

`method0 writes best, but it is not verifiable.`

Adding retrieval, citation requirements, and dependency planning appears to reduce surface-level writing quality, especially structure and organization.

### 3.2 Lexical Informativeness

| Method | ROUGE-L | METEOR |
|---|---:|---:|
| `method3` | `14.8578 ± 3.2536` | `19.6403 ± 9.4961` |
| `method2` | `14.6840 ± 3.8523` | `19.0072 ± 10.2247` |
| `method0` | `14.4014 ± 1.8387` | `17.8655 ± 6.4380` |
| `method1` | `13.5111 ± 6.5660` | `13.4096 ± 10.3597` |

`method3` has the highest average ROUGE-L and METEOR scores. This suggests that the Agent 2 + Agent 3 method produces articles that are lexically closest to the reference articles.

However, the margin between `method3` and `method2` is small:

| Metric | method3 | method2 |
|---|---:|---:|
| ROUGE-L | `14.8578` | `14.6840` |
| METEOR | `19.6403` | `19.0072` |

The lexical informativeness result therefore supports a cautious conclusion:

`method3 has the strongest average reference overlap, with method2 close behind.`

This suggests that dependency-aware generation may improve reference alignment, but the difference is modest.

### 3.3 Concept-Level Quality

| Method | Concept Score |
|---|---:|
| `method1` | `3.6500 ± 0.2687` |
| `method3` | `3.6500 ± 0.3764` |
| `method0` | `3.5750 ± 0.3736` |
| `method2` | `3.5500 ± 0.4830` |

`method1` and `method3` tie for the highest overall concept score.

The submetrics show different strengths:

| Metric | Best Method | Value |
|---|---|---:|
| Concept coverage | `method3` | `3.2000` |
| Concept accuracy | `method0` | `3.3000` |
| Concept relevance | `method1` | `4.8000` |
| Concept organization | `method1` | `3.9000` |

The concept evaluation shows that no method dominates all concept dimensions. `method3` has the best concept coverage, suggesting that dependency-aware generation helps include more important content. `method1` has the strongest concept relevance and organization, suggesting that naive RAG can remain focused and coherent at the concept level. `method0` has the highest concept accuracy, possibly because it produces more general or cautious content.

The main conclusion is:

`method3 improves concept coverage, but method1 remains equally strong in overall concept quality.`

### 3.4 Citation and Verifiability

| Method | Citation Rate | Citation Precision | Citation Recall |
|---|---:|---:|---:|
| `method1` | `0.9417 ± 0.1245` | `0.1817 ± 0.1596` | `0.1706 ± 0.1515` |
| `method2` | `0.8938 ± 0.1342` | `0.1777 ± 0.1523` | `0.1525 ± 0.1251` |
| `method3` | `0.8825 ± 0.0953` | `0.1964 ± 0.2499` | `0.1710 ± 0.2213` |
| `method0` | `0.0000 ± 0.0000` | `0.0000 ± 0.0000` | `0.0000 ± 0.0000` |

`method1` has the highest citation rate:

`method1 citation_rate = 0.9417`

This means it cites nearly every sentence. However, `method3` has the highest citation precision:

`method3 citation_precision = 0.1964`

and the highest citation recall:

`method3 citation_recall = 0.1710`

The differences between the RAG methods are small, and the standard deviations are large, especially for `method3`.

For example:

| Method | Citation Precision Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| `method3` | `0.1964` | `0.2499` | `0.0000` | `0.7143` |

This means `method3` sometimes produces much better citation support, but not consistently across all islands.

The citation results show an important RAG finding:

`citation presence is not the same as citation correctness.`

`method1` cites the most, but `method3` has the best average citation support. Still, all RAG methods have low citation precision overall. This means citation grounding remains a major bottleneck.

The main conclusion is:

`RAG methods increase citation coverage, but citation support remains weak. method3 shows the best average citation support, but the improvement is modest and variable.`

### 3.5 Cross-Sectional Consistency

| Method | CSCS |
|---|---:|
| `method3` | `0.5088 ± 0.0842` |
| `method0` | Not applicable |
| `method1` | Not applicable |
| `method2` | Not applicable |

Only `method3` has a CSCS score because only `method3` uses the Agent 2 dependency graph during generation.

The score is:

`method3 CSCS = 0.5088 ± 0.0842`

with:

| Min | Max |
|---:|---:|
| `0.3400` | `0.6268` |

This indicates moderate cross-sectional consistency.

The CSCS result suggests that `method3` partially preserves dependency information across sections. However, the score is not close to 1.0, so dependency preservation is incomplete.

The main conclusion is:

`method3 shows moderate cross-section dependency reflection, but there is still room for improvement.`

---

## 4. Method-Level Interpretation

### 4.1 Method0: Pure Generation Baseline

`method0` has the best writing quality:

| Metric | Value |
|---|---:|
| `writing_score` | `4.0650` |
| `fluency_score` | `4.4000` |

It also has reasonable lexical and concept scores:

| Metric | Value |
|---|---:|
| `ROUGE-L` | `14.4014` |
| `METEOR` | `17.8655` |
| `concept_score` | `3.5750` |

However, it has no citation support:

| Metric | Value |
|---|---:|
| `citation_rate` | `0.0000` |
| `citation_precision` | `0.0000` |
| `citation_recall` | `0.0000` |

`method0` is the strongest surface-writing baseline. It produces fluent and coherent prose, but it is not verifiable. Therefore, it cannot be considered a strong RAG method.

Conclusion for `method0`:

`method0 is best for writing quality, but weakest for grounded generation.`

### 4.2 Method1: Naive RAG

`method1` has a strong writing score:

`writing_score = 3.8340`

It also ties with `method3` for the best concept score:

`concept_score = 3.6500`

It has the highest citation rate:

`citation_rate = 0.9417`

However, its citation precision remains low:

`citation_precision = 0.1817`

`method1` successfully adds citations to most sentences, but many of these citations are not verified as supporting the sentence-level claims.

Conclusion for `method1`:

`method1 is good at citation coverage, but citation quality remains limited.`

### 4.3 Method2: Hierarchical RAG

`method2` improves lexical reference overlap compared with `method1`:

| Metric | Value |
|---|---:|
| `ROUGE-L` | `14.6840` |
| `METEOR` | `19.0072` |

However, it has the lowest writing score:

`writing_score = 3.4000`

Its citation precision is similar to `method1`:

`citation_precision = 0.1777`

`method2` appears to improve reference similarity, but this comes with weaker surface writing quality. Its hierarchical structure may help retrieve or organize relevant information, but this does not clearly translate into stronger citation support.

Conclusion for `method2`:

`method2 improves reference overlap, but it does not clearly improve writing or citation grounding.`

### 4.4 Method3: Agent 2 Plan + Agent 3 Context-Aware Generation

`method3` has the strongest lexical informativeness:

| Metric | Value |
|---|---:|
| `ROUGE-L` | `14.8578` |
| `METEOR` | `19.6403` |

It also ties for the best concept score:

`concept_score = 3.6500`

It has the best average citation precision and recall:

| Metric | Value |
|---|---:|
| `citation_precision` | `0.1964` |
| `citation_recall` | `0.1710` |

It is also the only method with CSCS:

`CSCS = 0.5088`

However, its writing score is lower than `method0` and `method1`:

`writing_score = 3.4660`

`method3` is not the most fluent method, but it performs best on several RAG-specific dimensions. It improves reference similarity, concept coverage, citation support, and cross-sectional consistency. However, citation precision remains low overall, and the high standard deviation shows that performance varies substantially across islands.

Conclusion for `method3`:

`method3 is the strongest RAG-oriented method, but not the strongest writing method.`

---

## 5. Overall Conclusion

The results show a clear trade-off between surface writing quality and grounded, dependency-aware generation.

`method0` produces the best prose, but it lacks citations and therefore lacks verifiability.

RAG-based methods substantially increase citation coverage, but citation coverage alone is not enough. The citation precision scores show that many cited sentences are still not strongly supported by the retrieved evidence.

Among the RAG methods, `method3` provides the strongest overall RAG-oriented performance. It achieves the highest average ROUGE-L, METEOR, citation precision, citation recall, and a moderate CSCS score. This suggests that Agent 2 dependency planning and Agent 3 context-aware generation improve reference alignment and cross-sectional consistency.

However, the improvements in citation support are modest and variable. Citation grounding remains the main weakness across all RAG methods.

The main finding can be summarized as:

```text
method0 writes best.
method1 cites most.
method2 improves reference overlap but weakens writing.
method3 performs best on RAG-specific dimensions, especially reference alignment,
citation support, and cross-sectional consistency.