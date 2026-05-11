"""
00_evaluation/error_analysis.py

Automatic error analysis over evaluation CSVs produced by full_evaluation.py
and combined by aggregate_evaluations.py.

Recommended input:
    00_evaluation/evaluations/aggregated/all_evaluations_combined.csv

This file should contain row-level outputs with columns such as:
    variant_name
    variant_type
    island_name
    method
    writing_score
    concept_score
    citation_precision
    cscs
    etc.

Outputs:
    error_instances_*.csv
    error_frequency_by_method_*.csv
    error_frequency_by_variant_*.csv
    error_frequency_by_variant_method_*.csv
    error_frequency_by_island_*.csv
    error_frequency_overall_*.csv
    top_error_subclasses_*.csv
    error_sensitivity_*.csv

Run from project root:

    python 00_evaluation/error_analysis.py ^
      --input 00_evaluation/evaluations/aggregated/all_evaluations_combined.csv ^
      --output-dir 00_evaluation/evaluations/error_analysis

If your aggregated folder is named 00_aggregated:

    python 00_evaluation/error_analysis.py ^
      --input 00_evaluation/evaluations/00_aggregated/all_evaluations_combined.csv ^
      --output-dir 00_evaluation/evaluations/error_analysis
"""

import argparse
import csv
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------
# Threshold settings
# ---------------------------------------------------------------------

THRESHOLD_SETTINGS = {
    # Fewer outputs are marked as errors.
    "lenient": {
        "medium_percentile": 0.20,
        "high_percentile": 0.05,
        "absolute_floor": 0.40,
    },

    # Main setting used in the paper/report.
    "main": {
        "medium_percentile": 0.25,
        "high_percentile": 0.10,
        "absolute_floor": 0.50,
    },

    # More outputs are marked as errors.
    "strict": {
        "medium_percentile": 0.33,
        "high_percentile": 0.15,
        "absolute_floor": 0.60,
    },
}


# Metrics where lower values mean worse quality and no universal cutoff exists.
# These are thresholded using global percentiles.
DISTRIBUTION_METRICS = [
    "rouge_l",
    "meteor",
    "citation_recall",
    "citation_precision",
    "citation_link_precision",
    "citation_rate",
    "cscs",
]

# Metric-specific absolute thresholds.
# These prevent one universal floor, e.g. 0.50, from being too harsh for
# metrics such as citation_link_precision.
METRIC_ABSOLUTE_THRESHOLDS = {
    "lenient": {
        "citation_rate": {"high": 0.30, "medium": 0.70},
        "citation_recall": {"high": 0.05, "medium": 0.15},
        "citation_precision": {"high": 0.05, "medium": 0.15},
        "citation_link_precision": {"high": 0.03, "medium": 0.07},
        "cscs": {"high": 0.35, "medium": 0.45},
    },

    "main": {
        "citation_rate": {"high": 0.50, "medium": 0.80},
        "citation_recall": {"high": 0.10, "medium": 0.20},
        "citation_precision": {"high": 0.10, "medium": 0.25},
        "citation_link_precision": {"high": 0.05, "medium": 0.10},
        "cscs": {"high": 0.40, "medium": 0.50},
    },

    "strict": {
        "citation_rate": {"high": 0.70, "medium": 0.90},
        "citation_recall": {"high": 0.15, "medium": 0.30},
        "citation_precision": {"high": 0.15, "medium": 0.30},
        "citation_link_precision": {"high": 0.08, "medium": 0.15},
        "cscs": {"high": 0.45, "medium": 0.55},
    },
}


# Rubric-based 1-5 evaluator scores.
RUBRIC_THRESHOLDS = {
    "fluency_score": {
        "medium": 3.0,
        "high": 2.5,
        "family": "Writing quality",
        "class": "Writing error",
        "subclass": "Low fluency",
        "fix": "Improve sentence-level fluency and readability.",
    },
    "structure_score": {
        "medium": 3.0,
        "high": 2.5,
        "family": "Writing quality",
        "class": "Writing error",
        "subclass": "Poor article structure",
        "fix": "Improve section structure and article-level organization.",
    },
    "organization_score": {
        "medium": 3.0,
        "high": 2.5,
        "family": "Writing quality",
        "class": "Writing error",
        "subclass": "Poor organization",
        "fix": "Improve logical ordering and transitions between ideas.",
    },
    "writing_score": {
        "medium": 3.0,
        "high": 2.5,
        "family": "Writing quality",
        "class": "Writing error",
        "subclass": "Low overall writing quality",
        "fix": "Improve fluency, structure, and organization jointly.",
    },
    "concept_coverage_score": {
        "medium": 3.0,
        "high": 2.5,
        "family": "Content quality",
        "class": "Content omission",
        "subclass": "Low concept coverage",
        "fix": "Retrieve and include more reference-relevant concepts.",
    },
    "concept_accuracy_score": {
        # Still stricter than other concept dimensions, but not overly punitive.
        "medium": 3.5,
        "high": 3.0,
        "family": "Content quality",
        "class": "Content substitution",
        "subclass": "Low concept accuracy",
        "fix": "Reduce unsupported or incorrect claims and improve grounding.",
    },
    "concept_relevance_score": {
        "medium": 3.0,
        "high": 2.5,
        "family": "Content quality",
        "class": "Content addition",
        "subclass": "Low concept relevance",
        "fix": "Remove irrelevant or off-topic content.",
    },
    "concept_organization_score": {
        "medium": 3.0,
        "high": 2.5,
        "family": "Content quality",
        "class": "Content organization error",
        "subclass": "Poor concept organization",
        "fix": "Reorder concepts so related facts appear in coherent sections.",
    },
    "concept_score": {
        "medium": 3.0,
        "high": 2.5,
        "family": "Content quality",
        "class": "Content quality error",
        "subclass": "Low overall concept quality",
        "fix": "Improve concept coverage, accuracy, relevance, and organization.",
    },
}


# Mapping distribution metrics to error classes.
DISTRIBUTION_ERROR_MAP = {
    "rouge_l": {
        "family": "Content quality",
        "class": "Content omission",
        "subclass": "Low ROUGE-L",
        "fix": "Improve lexical and structural overlap with important reference content.",
    },
    "meteor": {
        "family": "Content quality",
        "class": "Content omission",
        "subclass": "Low METEOR",
        "fix": "Improve coverage of reference-relevant concepts and paraphrases.",
    },
    "citation_recall": {
        "family": "Citation and verifiability",
        "class": "Citation coverage failure",
        "subclass": "Low citation recall",
        "fix": "Increase support coverage for factual claims.",
    },
    "citation_precision": {
        "family": "Citation and verifiability",
        "class": "Citation support failure",
        "subclass": "Low citation precision",
        "fix": "Ensure cited sentences are actually supported by their cited evidence.",
    },
    "citation_link_precision": {
        "family": "Citation and verifiability",
        "class": "Citation support failure",
        "subclass": "Low citation link precision",
        "fix": "Improve source selection so citation links point to supporting evidence.",
    },
    "citation_rate": {
        "family": "Citation and verifiability",
        "class": "Citation coverage failure",
        "subclass": "Low citation rate",
        "fix": "Increase citation density for factual sentences.",
    },
    "cscs": {
        "family": "Planning and graph",
        "class": "Graph coherence failure",
        "subclass": "Low CSCS",
        "fix": "Improve support for dependency-linked facts across planned sections.",
    },
}


ERROR_INSTANCE_FIELDS = [
    "analysis_timestamp",
    "threshold_setting",
    "output_id",

    # Variant metadata
    "comparison_label",
    "variant_name",
    "variant_type",
    "eval_file",
    "eval_folder",

    # Source metadata
    "source_file",
    "input_json",
    "reference_file",
    "context_file",
    "plan_file",
    "island_name",
    "method",

    # Error metadata
    "error_family",
    "error_class",
    "error_subclass",
    "severity",
    "metric_name",
    "metric_value",
    "threshold_type",
    "threshold_value",
    "evidence",
    "suggested_fix",
]

FREQUENCY_FIELDS = [
    "threshold_setting",

    # Variant grouping
    "comparison_label",
    "variant_name",
    "variant_type",

    # Standard grouping
    "method",
    "island_name",

    # Error grouping
    "error_family",
    "error_class",
    "error_subclass",

    # Any-severity counts
    "total_error_instances",
    "affected_outputs",
    "total_outputs_in_group",
    "percentage_of_outputs",

    # Severity-specific output percentages
    "high_severity_affected_outputs",
    "high_severity_percentage_of_outputs",
    "medium_severity_affected_outputs",
    "medium_severity_percentage_of_outputs",
    "low_severity_affected_outputs",
    "low_severity_percentage_of_outputs",

    # Severity-specific instance counts
    "high_severity_count",
    "medium_severity_count",
    "low_severity_count",
]


DENOMINATOR_GROUP_FIELDS = [
    "comparison_label",
    "variant_name",
    "variant_type",
    "method",
    "island_name",
]


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    text = str(value).strip()

    if text == "":
        return None

    lowered = text.lower()
    if lowered in {"nan", "none", "null", "na", "n/a"}:
        return None

    try:
        return float(text)
    except Exception:
        return None


def is_nonempty(value: Any) -> bool:
    if value is None:
        return False

    text = str(value).strip()

    if text == "":
        return False

    if text.lower() in {"none", "null", "nan", "n/a", "[]"}:
        return False

    return True


def truthy(value: Any) -> bool:
    if value is None:
        return False

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
        "used",
        "enabled",
    }


def count_listish_items(value: Any) -> int:
    if not is_nonempty(value):
        return 0

    text = str(value).strip()

    # Most fields in full_evaluation.py are semicolon-joined.
    if ";" in text:
        return len([x for x in text.split(";") if x.strip()])

    # Fallback for comma-separated text.
    if "," in text:
        return len([x for x in text.split(",") if x.strip()])

    return 1


def percentile(values: List[float], q: float) -> Optional[float]:
    """
    Linear interpolation percentile.

    q should be in [0, 1].
    """
    clean_values = sorted(v for v in values if v is not None)

    if not clean_values:
        return None

    if len(clean_values) == 1:
        return clean_values[0]

    q = max(0.0, min(1.0, q))
    pos = q * (len(clean_values) - 1)
    lower = int(pos)
    upper = min(lower + 1, len(clean_values) - 1)
    weight = pos - lower

    return clean_values[lower] * (1 - weight) + clean_values[upper] * weight


def format_float(value: Optional[float]) -> str:
    if value is None:
        return ""

    return f"{value:.6g}"


def make_comparison_label_for_row(row: Dict[str, Any]) -> str:
    variant = str(row.get("variant_name", "")).strip()
    method = str(row.get("method", "")).strip()

    if variant.lower() == "default":
        return f"default_{method}"

    if variant:
        return variant

    return method


def make_output_id(row: Dict[str, Any], row_index: int) -> str:
    pieces = [
        str(row.get("comparison_label", "")).strip(),
        str(row.get("variant_name", "")).strip(),
        str(row.get("input_json", "")).strip(),
        str(row.get("source_file", "")).strip(),
        str(row.get("island_name", "")).strip(),
        str(row.get("method", "")).strip(),
        str(row_index),
    ]

    return " | ".join(pieces)


def load_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for i, row in enumerate(rows):
        row["variant_name"] = str(row.get("variant_name", "")).strip()
        row["variant_type"] = str(row.get("variant_type", "")).strip()
        row["method"] = str(row.get("method", "")).strip()
        row["island_name"] = str(row.get("island_name", "")).strip()

        row["comparison_label"] = make_comparison_label_for_row(row)

        row["__row_index"] = i
        row["__output_id"] = make_output_id(row, i)

    return rows


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def derive_output_tag(input_path: str) -> str:
    base = os.path.basename(input_path)
    stem, _ = os.path.splitext(base)

    if stem.startswith("full_evaluation_"):
        stem = stem.replace("full_evaluation_", "", 1)

    return stem


def get_output_paths(input_path: str, output_dir: Optional[str]) -> Dict[str, str]:
    if output_dir is None:
        output_dir = os.path.dirname(input_path)

    tag = derive_output_tag(input_path)

    return {
        "instances": os.path.join(output_dir, f"error_instances_{tag}.csv"),
        "by_method": os.path.join(output_dir, f"error_frequency_by_method_{tag}.csv"),
        "by_variant": os.path.join(output_dir, f"error_frequency_by_variant_{tag}.csv"),
        "by_variant_method": os.path.join(
            output_dir,
            f"error_frequency_by_variant_method_{tag}.csv",
        ),
        "by_island": os.path.join(output_dir, f"error_frequency_by_island_{tag}.csv"),
        "overall": os.path.join(output_dir, f"error_frequency_overall_{tag}.csv"),
        "top_subclasses": os.path.join(output_dir, f"top_error_subclasses_{tag}.csv"),
        "sensitivity": os.path.join(output_dir, f"error_sensitivity_{tag}.csv"),
    }


# ---------------------------------------------------------------------
# Threshold computation
# ---------------------------------------------------------------------

def metric_applicable(row: Dict[str, Any], metric: str) -> bool:
    method = str(row.get("method", "")).strip().lower()

    if metric == "cscs":
        return method == "method3" and safe_float(row.get("cscs")) is not None

    return True


def compute_distribution_thresholds(
    rows: List[Dict[str, Any]],
    setting: Dict[str, float],
    min_percentile_n: int,
) -> Dict[str, Dict[str, Any]]:
    thresholds = {}

    for metric in DISTRIBUTION_METRICS:
        values = []

        for row in rows:
            if not metric_applicable(row, metric):
                continue

            value = safe_float(row.get(metric))

            if value is not None:
                values.append(value)

        if len(values) < min_percentile_n:
            thresholds[metric] = {
                "n": len(values),
                "medium": None,
                "high": None,
            }
            continue

        thresholds[metric] = {
            "n": len(values),
            "medium": percentile(values, setting["medium_percentile"]),
            "high": percentile(values, setting["high_percentile"]),
        }

    return thresholds


# ---------------------------------------------------------------------
# Error construction
# ---------------------------------------------------------------------

def add_error(
    errors: List[Dict[str, Any]],
    row: Dict[str, Any],
    threshold_setting: str,
    error_family: str,
    error_class: str,
    error_subclass: str,
    severity: str,
    metric_name: str,
    metric_value: Any,
    threshold_type: str,
    threshold_value: Any,
    evidence: str,
    suggested_fix: str,
) -> None:
    errors.append({
        "analysis_timestamp": datetime.now().isoformat(timespec="seconds"),
        "threshold_setting": threshold_setting,
        "output_id": row.get("__output_id", ""),

        # Variant metadata
        "comparison_label": row.get("comparison_label", ""),
        "variant_name": row.get("variant_name", ""),
        "variant_type": row.get("variant_type", ""),
        "eval_file": row.get("eval_file", ""),
        "eval_folder": row.get("eval_folder", ""),

        # Source metadata
        "source_file": row.get("source_file", ""),
        "input_json": row.get("input_json", ""),
        "reference_file": row.get("reference_file", ""),
        "context_file": row.get("context_file", ""),
        "plan_file": row.get("plan_file", ""),
        "island_name": row.get("island_name", ""),
        "method": row.get("method", ""),

        # Error metadata
        "error_family": error_family,
        "error_class": error_class,
        "error_subclass": error_subclass,
        "severity": severity,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "threshold_type": threshold_type,
        "threshold_value": threshold_value,
        "evidence": evidence,
        "suggested_fix": suggested_fix,
    })


def classify_rubric_errors(
    row: Dict[str, Any],
    errors: List[Dict[str, Any]],
    threshold_setting: str,
) -> None:
    for metric, spec in RUBRIC_THRESHOLDS.items():
        value = safe_float(row.get(metric))

        if value is None:
            continue

        severity = None
        threshold_value = None

        if value < spec["high"]:
            severity = "high"
            threshold_value = spec["high"]
        elif value < spec["medium"]:
            severity = "medium"
            threshold_value = spec["medium"]

        if severity:
            add_error(
                errors=errors,
                row=row,
                threshold_setting=threshold_setting,
                error_family=spec["family"],
                error_class=spec["class"],
                error_subclass=spec["subclass"],
                severity=severity,
                metric_name=metric,
                metric_value=format_float(value),
                threshold_type="rubric",
                threshold_value=f"< {threshold_value}",
                evidence=f"{metric} = {format_float(value)}",
                suggested_fix=spec["fix"],
            )

def classify_distribution_metric(
    row: Dict[str, Any],
    metric: str,
    thresholds: Dict[str, Dict[str, Any]],
    setting: Dict[str, float],
    threshold_setting: str,
    errors: List[Dict[str, Any]],
) -> None:
    if metric not in DISTRIBUTION_ERROR_MAP:
        return

    if not metric_applicable(row, metric):
        return

    value = safe_float(row.get(metric))

    if value is None:
        return

    metric_thresholds = thresholds.get(metric, {})
    high_threshold = metric_thresholds.get("high")
    medium_threshold = metric_thresholds.get("medium")

    floor_spec = METRIC_ABSOLUTE_THRESHOLDS.get(
        threshold_setting,
        {},
    ).get(metric, {})

    absolute_high_floor = floor_spec.get("high")
    absolute_medium_floor = floor_spec.get("medium")

    high_hit = False
    medium_hit = False
    threshold_parts = []

    # Percentile-based thresholding:
    # marks the lower tail of the observed distribution.
    if high_threshold is not None:
        threshold_parts.append(
            f"high_percentile <= {format_float(high_threshold)}"
        )

        if value <= high_threshold:
            high_hit = True

    if medium_threshold is not None:
        threshold_parts.append(
            f"medium_percentile <= {format_float(medium_threshold)}"
        )

        if value <= medium_threshold:
            medium_hit = True

    # Metric-specific absolute floors:
    # avoids using one universal 0.50 floor for all bounded metrics.
    if absolute_high_floor is not None:
        threshold_parts.append(
            f"absolute_high_floor < {format_float(absolute_high_floor)}"
        )

        if value < absolute_high_floor:
            high_hit = True

    if absolute_medium_floor is not None:
        threshold_parts.append(
            f"absolute_medium_floor < {format_float(absolute_medium_floor)}"
        )

        if value < absolute_medium_floor:
            medium_hit = True

    if high_hit:
        severity = "high"
    elif medium_hit:
        severity = "medium"
    else:
        return

    spec = DISTRIBUTION_ERROR_MAP[metric]

    threshold_type = "distribution"

    if absolute_high_floor is not None or absolute_medium_floor is not None:
        threshold_type = "distribution_or_metric_specific_floor"

    add_error(
        errors=errors,
        row=row,
        threshold_setting=threshold_setting,
        error_family=spec["family"],
        error_class=spec["class"],
        error_subclass=spec["subclass"],
        severity=severity,
        metric_name=metric,
        metric_value=format_float(value),
        threshold_type=threshold_type,
        threshold_value="; ".join(threshold_parts),
        evidence=f"{metric} = {format_float(value)}",
        suggested_fix=spec["fix"],
    )


def classify_distribution_errors(
    row: Dict[str, Any],
    errors: List[Dict[str, Any]],
    thresholds: Dict[str, Dict[str, Any]],
    setting: Dict[str, float],
    threshold_setting: str,
) -> None:
    for metric in DISTRIBUTION_METRICS:
        classify_distribution_metric(
            row=row,
            metric=metric,
            thresholds=thresholds,
            setting=setting,
            threshold_setting=threshold_setting,
            errors=errors,
        )



def classify_categorical_errors(
    row: Dict[str, Any],
    errors: List[Dict[str, Any]],
    threshold_setting: str,
) -> None:
    # -------------------------------------------------------------
    # Content omission
    # -------------------------------------------------------------
    if is_nonempty(row.get("missing_key_concepts")):
        count = count_listish_items(row.get("missing_key_concepts"))
        severity = "high" if count >= 2 else "medium"

        add_error(
            errors,
            row,
            threshold_setting,
            "Content quality",
            "Content omission",
            "Missing key concepts",
            severity,
            "missing_key_concepts",
            count,
            "categorical",
            "non-empty",
            row.get("missing_key_concepts", ""),
            "Improve retrieval and planning so major reference concepts are included.",
        )

    # -------------------------------------------------------------
    # Content substitution / unsupported concepts
    # -------------------------------------------------------------
    if is_nonempty(row.get("inaccurate_or_unsupported_concepts")):
        count = count_listish_items(row.get("inaccurate_or_unsupported_concepts"))
        severity = "high" if count >= 1 else "medium"

        add_error(
            errors,
            row,
            threshold_setting,
            "Content quality",
            "Content substitution",
            "Inaccurate or unsupported concepts",
            severity,
            "inaccurate_or_unsupported_concepts",
            count,
            "categorical",
            "non-empty",
            row.get("inaccurate_or_unsupported_concepts", ""),
            "Add stronger grounding checks before final article generation.",
        )




def classify_row(
    row: Dict[str, Any],
    thresholds: Dict[str, Dict[str, Any]],
    setting: Dict[str, float],
    threshold_setting: str,
) -> List[Dict[str, Any]]:
    errors = []

    # Main article-quality errors only.
    # Artifact/pipeline diagnostics are intentionally excluded from the
    # primary error analysis.
    classify_categorical_errors(row, errors, threshold_setting)
    classify_rubric_errors(row, errors, threshold_setting)
    classify_distribution_errors(row, errors, thresholds, setting, threshold_setting)

    return errors


def classify_all_rows(
    rows: List[Dict[str, Any]],
    threshold_setting: str,
    min_percentile_n: int,
) -> List[Dict[str, Any]]:
    setting = THRESHOLD_SETTINGS[threshold_setting]

    thresholds = compute_distribution_thresholds(
        rows=rows,
        setting=setting,
        min_percentile_n=min_percentile_n,
    )

    errors = []

    for row in rows:
        row_errors = classify_row(
            row=row,
            thresholds=thresholds,
            setting=setting,
            threshold_setting=threshold_setting,
        )
        errors.extend(row_errors)

    return errors


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------

def denominator_fields_for_group_keys(group_keys: List[str]) -> List[str]:
    return [key for key in group_keys if key in DENOMINATOR_GROUP_FIELDS]


def build_denominators(
    rows: List[Dict[str, Any]],
    denominator_fields: List[str],
) -> Dict[Tuple[Any, ...], int]:
    denominators = defaultdict(set)

    for row in rows:
        output_id = row.get("__output_id", "")
        key = tuple(row.get(field, "") for field in denominator_fields)
        denominators[key].add(output_id)

    return {key: len(value) for key, value in denominators.items()}


def aggregate_errors(
    errors: List[Dict[str, Any]],
    rows: List[Dict[str, Any]],
    group_keys: List[str],
) -> List[Dict[str, Any]]:
    denominator_fields = denominator_fields_for_group_keys(group_keys)
    denominators = build_denominators(rows, denominator_fields)

    grouped = {}

    for error in errors:
        key = tuple(error.get(k, "") for k in group_keys)

        if key not in grouped:
            group_row = {k: error.get(k, "") for k in group_keys}
            group_row["total_error_instances"] = 0
            group_row["affected_output_ids"] = set()

            # Unique affected outputs by severity
            group_row["high_severity_output_ids"] = set()
            group_row["medium_severity_output_ids"] = set()
            group_row["low_severity_output_ids"] = set()

            # Error-instance counts by severity
            group_row["high_severity_count"] = 0
            group_row["medium_severity_count"] = 0
            group_row["low_severity_count"] = 0
            grouped[key] = group_row

        group = grouped[key]
        group["total_error_instances"] += 1
        group["affected_output_ids"].add(error.get("output_id", ""))

        severity = str(error.get("severity", "")).strip().lower()
        output_id = error.get("output_id", "")

        if severity == "high":
            group["high_severity_count"] += 1
            group["high_severity_output_ids"].add(output_id)

        elif severity == "medium":
            group["medium_severity_count"] += 1
            group["medium_severity_output_ids"].add(output_id)

        elif severity == "low":
            group["low_severity_count"] += 1
            group["low_severity_output_ids"].add(output_id)

    output = []

    for group in grouped.values():
        affected_outputs = len(group["affected_output_ids"])

        high_affected_outputs = len(group["high_severity_output_ids"])
        medium_affected_outputs = len(group["medium_severity_output_ids"])
        low_affected_outputs = len(group["low_severity_output_ids"])

        denominator_key = tuple(group.get(field, "") for field in denominator_fields)
        total_outputs = denominators.get(denominator_key, len(rows))

        if total_outputs:
            percentage = affected_outputs / total_outputs
            high_percentage = high_affected_outputs / total_outputs
            medium_percentage = medium_affected_outputs / total_outputs
            low_percentage = low_affected_outputs / total_outputs
        else:
            percentage = 0.0
            high_percentage = 0.0
            medium_percentage = 0.0
            low_percentage = 0.0

        row = {
            "threshold_setting": group.get("threshold_setting", ""),
            "comparison_label": group.get("comparison_label", ""),
            "variant_name": group.get("variant_name", ""),
            "variant_type": group.get("variant_type", ""),
            "method": group.get("method", ""),
            "island_name": group.get("island_name", ""),
            "error_family": group.get("error_family", ""),
            "error_class": group.get("error_class", ""),
            "error_subclass": group.get("error_subclass", ""),
            "total_error_instances": group["total_error_instances"],
            "affected_outputs": affected_outputs,
            "total_outputs_in_group": total_outputs,
            "percentage_of_outputs": f"{percentage:.6f}",

            "high_severity_affected_outputs": high_affected_outputs,
            "high_severity_percentage_of_outputs": f"{high_percentage:.6f}",
            "medium_severity_affected_outputs": medium_affected_outputs,
            "medium_severity_percentage_of_outputs": f"{medium_percentage:.6f}",
            "low_severity_affected_outputs": low_affected_outputs,
            "low_severity_percentage_of_outputs": f"{low_percentage:.6f}",

            "high_severity_count": group["high_severity_count"],
            "medium_severity_count": group["medium_severity_count"],
            "low_severity_count": group["low_severity_count"],
        }

        output.append(row)

    output.sort(
        key=lambda r: (
            r.get("threshold_setting", ""),
            r.get("comparison_label", ""),
            r.get("variant_name", ""),
            r.get("method", ""),
            r.get("island_name", ""),
            r.get("error_family", ""),
            r.get("error_class", ""),
            -int(r.get("total_error_instances", 0)),
        )
    )

    return output


def top_error_subclasses(
    errors: List[Dict[str, Any]],
    rows: List[Dict[str, Any]],
    top_n: int,
) -> List[Dict[str, Any]]:
    aggregated = aggregate_errors(
        errors=errors,
        rows=rows,
        group_keys=[
            "threshold_setting",
            "error_family",
            "error_class",
            "error_subclass",
        ],
    )

    aggregated.sort(
        key=lambda r: (
            -int(r.get("total_error_instances", 0)),
            r.get("error_family", ""),
            r.get("error_class", ""),
            r.get("error_subclass", ""),
        )
    )

    return aggregated[:top_n]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Automatic error analysis over full_evaluation.py CSV output."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to row-level evaluation CSV, usually all_evaluations_combined.csv.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for error-analysis CSV outputs. Defaults to input CSV directory.",
    )

    parser.add_argument(
        "--main-setting",
        choices=["lenient", "main", "strict"],
        default="main",
        help="Threshold setting used for primary error tables.",
    )

    parser.add_argument(
        "--min-percentile-n",
        type=int,
        default=5,
        help=(
            "Minimum number of numeric values required to compute percentile thresholds "
            "for a metric. Default: 5."
        ),
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Number of top error subclasses to write. Default: 25.",
    )

    parser.add_argument(
        "--no-sensitivity",
        action="store_true",
        help="Disable lenient/main/strict sensitivity output.",
    )

    args = parser.parse_args()

    rows = load_csv(args.input)

    if not rows:
        raise ValueError(f"No rows found in input CSV: {args.input}")

    output_paths = get_output_paths(args.input, args.output_dir)

    print("\n" + "=" * 80)
    print("Automatic Error Analysis")
    print("=" * 80)
    print(f"Input CSV:          {args.input}")
    print(f"Rows:               {len(rows)}")
    print(f"Main setting:       {args.main_setting}")
    print(f"Min percentile n:   {args.min_percentile_n}")
    print(f"Output directory:   {os.path.dirname(output_paths['instances'])}")
    print("=" * 80 + "\n")

    # -------------------------------------------------------------
    # Main error analysis
    # -------------------------------------------------------------
    main_errors = classify_all_rows(
        rows=rows,
        threshold_setting=args.main_setting,
        min_percentile_n=args.min_percentile_n,
    )

    write_csv(
        output_paths["instances"],
        main_errors,
        ERROR_INSTANCE_FIELDS,
    )

    by_method = aggregate_errors(
        errors=main_errors,
        rows=rows,
        group_keys=[
            "threshold_setting",
            "method",
            "error_family",
            "error_class",
        ],
    )

    write_csv(
        output_paths["by_method"],
        by_method,
        FREQUENCY_FIELDS,
    )

    by_variant = aggregate_errors(
        errors=main_errors,
        rows=rows,
        group_keys=[
            "threshold_setting",
            "comparison_label",
            "variant_name",
            "variant_type",
            "error_family",
            "error_class",
        ],
    )

    write_csv(
        output_paths["by_variant"],
        by_variant,
        FREQUENCY_FIELDS,
    )

    by_variant_method = aggregate_errors(
        errors=main_errors,
        rows=rows,
        group_keys=[
            "threshold_setting",
            "comparison_label",
            "variant_name",
            "variant_type",
            "method",
            "error_family",
            "error_class",
        ],
    )

    write_csv(
        output_paths["by_variant_method"],
        by_variant_method,
        FREQUENCY_FIELDS,
    )

    by_island = aggregate_errors(
        errors=main_errors,
        rows=rows,
        group_keys=[
            "threshold_setting",
            "island_name",
            "error_family",
            "error_class",
        ],
    )

    write_csv(
        output_paths["by_island"],
        by_island,
        FREQUENCY_FIELDS,
    )

    overall = aggregate_errors(
        errors=main_errors,
        rows=rows,
        group_keys=[
            "threshold_setting",
            "error_family",
            "error_class",
        ],
    )

    write_csv(
        output_paths["overall"],
        overall,
        FREQUENCY_FIELDS,
    )

    top_subclasses = top_error_subclasses(
        errors=main_errors,
        rows=rows,
        top_n=args.top_n,
    )

    write_csv(
        output_paths["top_subclasses"],
        top_subclasses,
        FREQUENCY_FIELDS,
    )

    # -------------------------------------------------------------
    # Sensitivity analysis
    # -------------------------------------------------------------
    if not args.no_sensitivity:
        sensitivity_errors = []

        for setting_name in ["lenient", "main", "strict"]:
            setting_errors = classify_all_rows(
                rows=rows,
                threshold_setting=setting_name,
                min_percentile_n=args.min_percentile_n,
            )
            sensitivity_errors.extend(setting_errors)

        sensitivity = aggregate_errors(
            errors=sensitivity_errors,
            rows=rows,
            group_keys=[
                "threshold_setting",
                "method",
                "error_family",
                "error_class",
            ],
        )

        write_csv(
            output_paths["sensitivity"],
            sensitivity,
            FREQUENCY_FIELDS,
        )

    print("Saved:")
    print(f"  Error instances:              {output_paths['instances']}")
    print(f"  Frequency by method:          {output_paths['by_method']}")
    print(f"  Frequency by variant:         {output_paths['by_variant']}")
    print(f"  Frequency by variant/method:  {output_paths['by_variant_method']}")
    print(f"  Frequency by island:          {output_paths['by_island']}")
    print(f"  Overall frequency:            {output_paths['overall']}")
    print(f"  Top error subclasses:         {output_paths['top_subclasses']}")

    if not args.no_sensitivity:
        print(f"  Sensitivity analysis:         {output_paths['sensitivity']}")

    print("\nDone.")


if __name__ == "__main__":
    main()