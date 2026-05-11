# Results Summary

This document explains the aggregated evaluation results.

The results are organized in two parts:

1. **Default method comparison:** how `method0`, `method1`, `method2`, and `method3` perform.
2. **Method 3 ablation comparison:** how different Method 3 variants perform against the default Method 3 setup.

---

# 1. Default Method Comparison

Source file:

```text
00_evaluation/evaluations/00_aggregated/default_by_method_pretty.csv
```

Each method is evaluated on:

```text
10 islands
```

The table reports mean ± standard deviation.

---

## 1.1 Overall default method results

| Method  |         Writing |          ROUGE-L |           METEOR |   Concept score | Citation precision | Citation link precision |   Citation rate |            CSCS |
| ------- | --------------: | ---------------: | ---------------: | --------------: | -----------------: | ----------------------: | --------------: | --------------: |
| method0 | 4.1990 ± 0.3204 | 15.7924 ± 2.3616 | 26.3162 ± 4.1166 | 4.0500 ± 0.3496 |    0.0000 ± 0.0000 |         0.0000 ± 0.0000 | 0.0000 ± 0.0000 |               — |
| method1 | 3.9010 ± 0.1594 | 13.8597 ± 3.3729 | 16.0993 ± 9.1728 | 3.7250 ± 0.3623 |    0.2330 ± 0.2035 |         0.1049 ± 0.1171 | 0.9789 ± 0.0446 |               — |
| method2 | 3.4330 ± 0.2274 | 16.2569 ± 2.5955 | 22.1475 ± 8.0204 | 3.5750 ± 0.5144 |    0.1938 ± 0.1600 |         0.0699 ± 0.0673 | 0.9408 ± 0.1010 |               — |
| method3 | 3.6000 ± 0.3454 | 15.6986 ± 2.5423 | 22.6172 ± 9.2699 | 3.7000 ± 0.4534 |    0.2034 ± 0.1913 |         0.0612 ± 0.0682 | 0.9746 ± 0.0541 | 0.5073 ± 0.0757 |

---

## 1.2 Interpretation of default methods

### Writing quality

`method0` performs best on writing quality:

```text
method0 writing_score = 4.1990 ± 0.3204
```

This means the pure-generation baseline produces the most fluent and polished articles. Among the retrieval/planning methods, `method1` has the strongest writing score.

---

### Informativeness with ROUGE-L and METEOR

For ROUGE-L, `method2` performs best:

```text
method2 ROUGE-L = 16.2569 ± 2.5955
```

For METEOR, `method0` performs best:

```text
method0 METEOR = 26.3162 ± 4.1166
```

`method3` is close to `method2` on lexical informativeness, especially METEOR.

---

### Concept quality

`method0` has the highest overall concept score:

```text
method0 concept_score = 4.0500 ± 0.3496
```

Among the retrieval/planning methods, `method1` and `method3` are close:

```text
method1 concept_score = 3.7250 ± 0.3623
method3 concept_score = 3.7000 ± 0.4534
```

However, `method3` has the best concept coverage and concept accuracy among the retrieval/planning methods:

```text
method3 concept_coverage_score = 3.3000 ± 0.4830
method3 concept_accuracy_score = 3.6000 ± 0.5164
```

This suggests that the Agent 2 plan and context-aware generation help improve semantic coverage and accuracy.

---

### Citation quality

`method0` has zero citation scores because it is not citation-grounded in the same way as the RAG-based methods.

Among `method1`, `method2`, and `method3`, citation rates are high:

```text
method1 citation_rate = 0.9789 ± 0.0446
method3 citation_rate = 0.9746 ± 0.0541
method2 citation_rate = 0.9408 ± 0.1010
```

This means the systems place citations frequently.

However, citation precision and citation link precision are low:

```text
method1 citation_precision = 0.2330 ± 0.2035
method3 citation_precision = 0.2034 ± 0.1913
method2 citation_precision = 0.1938 ± 0.1600
```

```text
method1 citation_link_precision = 0.1049 ± 0.1171
method2 citation_link_precision = 0.0699 ± 0.0673
method3 citation_link_precision = 0.0612 ± 0.0682
```

This shows that the models often place citation markers, but many citations are not verified as supporting the cited claims.

---

### CSCS

CSCS is only computed for `method3` because only Method 3 uses the Agent 2 dependency graph.

```text
method3 CSCS = 0.5073 ± 0.0757
```

This indicates moderate cross-sectional consistency: some dependency information is preserved across sections, but there is room for improvement.

---

## 1.3 Summary of default method comparison

| Question                                           | Result         |
| -------------------------------------------------- | -------------- |
| Best writing quality                               | `method0`      |
| Best ROUGE-L                                       | `method2`      |
| Best METEOR                                        | `method0`      |
| Best overall concept score                         | `method0`      |
| Best concept coverage among RAG/planning methods   | `method3`      |
| Best concept accuracy among RAG/planning methods   | `method3`      |
| Best citation precision among RAG/planning methods | `method1`      |
| Method with CSCS                                   | `method3` only |

Main takeaway:

> `method0` produces the most polished articles, but it is not citation-grounded. Among the retrieval/planning methods, `method3` improves concept coverage and accuracy, while citation grounding remains weak across all RAG-based methods.

---

# 2. Method 3 Ablation Results

Source files:

```text
00_evaluation/evaluations/00_aggregated/method3_by_variant_pretty.csv
00_evaluation/evaluations/00_aggregated/method3_with_default_by_variant_pretty.csv
```

Each Method 3 variant is evaluated on:

```text
10 islands
```

The goal is to understand which Method 3 configuration improves which metric group.

---

## 2.1 Method 3 baseline

The default Method 3 row is:

| Variant         |         Writing |          ROUGE-L |           METEOR |   Concept score | Citation precision | Citation link precision |   Citation rate |            CSCS |
| --------------- | --------------: | ---------------: | ---------------: | --------------: | -----------------: | ----------------------: | --------------: | --------------: |
| default_method3 | 3.6000 ± 0.3454 | 15.6986 ± 2.5423 | 22.6172 ± 9.2699 | 3.7000 ± 0.4534 |    0.2034 ± 0.1913 |         0.0612 ± 0.0682 | 0.9746 ± 0.0541 | 0.5073 ± 0.0757 |

The ablations should be interpreted relative to this default Method 3 baseline.

---

## 2.2 Best Method 3 variants by metric

| Metric group            | Best variant                        |           Result |
| ----------------------- | ----------------------------------- | ---------------: |
| Writing score           | `threshold 0.6`                     |  3.6990 ± 0.3676 |
| ROUGE-L                 | `agent3 model claude opus 4.7`      | 17.1820 ± 2.9137 |
| METEOR                  | `agent3 model claude opus 4.7`      | 28.0961 ± 6.5702 |
| Concept score           | `agent2 model qwen3.6 plus`         |  3.8000 ± 0.3873 |
| Citation recall         | `reranker_type cross encoder + mmr` |  0.3444 ± 0.1986 |
| Citation precision      | `reranker_type cross encoder + mmr` |  0.3620 ± 0.2040 |
| Citation link precision | `reranker_type cross encoder + mmr` |  0.1825 ± 0.1345 |
| CSCS                    | `agent3 model gpt5.4`               |  0.5325 ± 0.0677 |

---

## 2.3 Method 3 ablation table

| Variant                             |         Writing |          ROUGE-L |           METEOR |   Concept score | Citation precision | Citation link precision |   Citation rate |            CSCS |
| ----------------------------------- | --------------: | ---------------: | ---------------: | --------------: | -----------------: | ----------------------: | --------------: | --------------: |
| default_method3                     | 3.6000 ± 0.3454 | 15.6986 ± 2.5423 | 22.6172 ± 9.2699 | 3.7000 ± 0.4534 |    0.2034 ± 0.1913 |         0.0612 ± 0.0682 | 0.9746 ± 0.0541 | 0.5073 ± 0.0757 |
| agent2 model deepseek v3.2          | 3.6330 ± 0.3322 | 15.8447 ± 2.8777 | 22.2636 ± 8.4544 | 3.6250 ± 0.4449 |    0.1358 ± 0.1083 |         0.0570 ± 0.0567 | 0.9574 ± 0.0768 | 0.5131 ± 0.0562 |
| agent2 model gemini 3 flash preview | 3.5670 ± 0.3175 | 15.7839 ± 2.8467 | 22.2771 ± 8.8469 | 3.6500 ± 0.4595 |    0.1792 ± 0.2186 |         0.0674 ± 0.0837 | 0.9612 ± 0.0693 | 0.5042 ± 0.0784 |
| agent2 model qwen3.6 plus           | 3.5660 ± 0.2764 | 15.6343 ± 2.6837 | 22.4771 ± 8.6431 | 3.8000 ± 0.3873 |    0.2319 ± 0.1925 |         0.0799 ± 0.0585 | 0.9641 ± 0.0634 | 0.4937 ± 0.0669 |
| agent3 model claude opus 4.7        | 3.2650 ± 0.3797 | 17.1820 ± 2.9137 | 28.0961 ± 6.5702 | 3.5000 ± 0.4410 |    0.2394 ± 0.2037 |         0.0879 ± 0.0897 | 0.9243 ± 0.0994 | 0.5250 ± 0.1122 |
| agent3 model gemini 3.1 pro preview | 3.3330 ± 0.3147 | 16.3843 ± 3.2327 | 22.3543 ± 8.3951 | 3.5000 ± 0.4249 |    0.1117 ± 0.1381 |         0.0496 ± 0.0626 | 0.9695 ± 0.0558 | 0.4778 ± 0.0963 |
| agent3 model gpt5.4                 | 3.6330 ± 0.2935 | 15.9739 ± 2.0528 | 23.9922 ± 7.2755 | 3.6500 ± 0.5297 |    0.1542 ± 0.1793 |         0.0610 ± 0.1033 | 0.9527 ± 0.0869 | 0.5325 ± 0.0677 |
| rerank_scope global                 | 3.5990 ± 0.3451 | 15.4988 ± 2.6786 | 20.4043 ± 7.8611 | 3.6750 ± 0.4257 |    0.1583 ± 0.1952 |         0.0692 ± 0.0909 | 0.9652 ± 0.0736 | 0.4990 ± 0.0626 |
| reranker_type cross encoder + mmr   | 3.6320 ± 0.4294 | 16.1587 ± 2.7956 | 22.5790 ± 8.4242 | 3.7250 ± 0.4632 |    0.3620 ± 0.2040 |         0.1825 ± 0.1345 | 0.9591 ± 0.0740 | 0.5066 ± 0.0943 |
| threshold 0.3                       | 3.5670 ± 0.2274 | 16.0616 ± 2.4707 | 22.2655 ± 8.2031 | 3.7000 ± 0.4048 |    0.2608 ± 0.2168 |         0.0799 ± 0.0795 | 0.9522 ± 0.0798 | 0.4964 ± 0.0986 |
| threshold 0.6                       | 3.6990 ± 0.3676 | 15.9496 ± 2.7210 | 22.8560 ± 8.8544 | 3.6750 ± 0.4091 |    0.2113 ± 0.1487 |         0.1163 ± 0.1113 | 0.9436 ± 0.0938 | 0.5246 ± 0.0785 |

---

## 2.4 Interpretation of Method 3 ablations

### Writing quality

The best Method 3 writing score comes from:

```text
threshold 0.6 = 3.6990 ± 0.3676
```

This is higher than the default Method 3 writing score:

```text
default_method3 = 3.6000 ± 0.3454
```

Interpretation:

> Increasing the threshold to 0.6 improves overall writing quality for Method 3.

---

### Lexical informativeness

The strongest lexical overlap comes from:

```text
agent3 model claude opus 4.7
ROUGE-L = 17.1820 ± 2.9137
METEOR = 28.0961 ± 6.5702
```

Interpretation:

> The Agent 3 generation model has a clear effect on how closely the generated article matches the reference article.

---

### Concept quality

The strongest concept score comes from:

```text
agent2 model qwen3.6 plus = 3.8000 ± 0.3873
```

This is higher than default Method 3:

```text
default_method3 = 3.7000 ± 0.4534
```

Interpretation:

> The Agent 2 planning model affects the semantic quality of the generated article, especially concept selection and organization.

---

### Citation and verifiability

The strongest citation performance comes from:

```text
reranker_type cross encoder + mmr
citation_recall = 0.3444 ± 0.1986
citation_precision = 0.3620 ± 0.2040
citation_link_precision = 0.1825 ± 0.1345
```

Compared with default Method 3:

```text
default citation_recall = 0.2021 ± 0.1920
default citation_precision = 0.2034 ± 0.1913
default citation_link_precision = 0.0612 ± 0.0682
```

Interpretation:

> Cross-encoder + MMR reranking substantially improves citation grounding. This is the strongest ablation result for verifiability.

---

### CSCS

The highest CSCS comes from:

```text
agent3 model gpt5.4 = 0.5325 ± 0.0677
```

Default Method 3 has:

```text
default_method3 = 0.5073 ± 0.0757
```

Interpretation:

> The Agent 3 generation model affects how well the final article preserves Agent 2 dependency information across sections.

---

# 3. Overall Interpretation

The results show a tradeoff between surface quality, semantic content quality, citation grounding, and dependency consistency.

## Main conclusions

1. **`method0` is the strongest for writing and overall concept score**, but it is not citation-grounded.

2. **`method3` is useful among retrieval/planning methods because it improves concept coverage and concept accuracy.**

3. **Citation grounding is the weakest part of the system.** All RAG-based methods place citations frequently, but citation precision and citation link precision remain low.

4. **For Method 3, reranking matters most for citation quality.** The cross-encoder + MMR variant gives the best citation recall, precision, and link precision.

5. **Agent 3 model choice affects both lexical overlap and dependency consistency.** Claude Opus 4.7 gives the best ROUGE-L/METEOR, while GPT-5.4 gives the best CSCS.

6. **No single variant wins across all metrics.** Different components improve different parts of the pipeline.

---

# 4. Short Final Summary

The default comparison shows that `method0` produces the most polished text, while `method3` provides the most relevant planning-based signal among retrieval methods, especially for concept coverage and accuracy. However, citation support remains weak across all citation-based methods.

The Method 3 ablation results show that the most useful improvement for verifiability is `reranker_type cross encoder + mmr`, while Agent 3 model choice has the strongest impact on lexical overlap and cross-sectional consistency.
