"""Export the frozen attachment-4 JSONL predictions as a flat submission CSV.

The CSV has one row per sample. Feature and character offsets are zero-based;
``*_end_exclusive`` is not part of the selected span. Empty media fields mean
that no defensible time/frame mapping is available. ``media_candidate`` is an
ASR-assisted candidate, not a verified feature-to-media provenance mapping.
"""

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/attachment4/predictions_20.jsonl"
OUTPUT = ROOT / "outputs/attachment4/attachment4_predictions_explanations.csv"
MODALITIES = ("text", "audio", "vision")
LABELS = ("Negative", "Neutral", "Positive")


def _location_fields(prefix, evidence):
    fields = {
        f"{prefix}_feature_start": "",
        f"{prefix}_feature_end_exclusive": "",
        f"{prefix}_decision_support": "",
        f"{prefix}_class_changed": "",
        f"{prefix}_mapping_status": "",
        f"{prefix}_raw_text_fragment": "",
        f"{prefix}_aligned_transcript_fragment": "",
        f"{prefix}_media_start_seconds": "",
        f"{prefix}_media_end_seconds": "",
        f"{prefix}_video_frame_index": "",
        f"{prefix}_video_frame_time_seconds": "",
    }
    if evidence is None:
        return fields
    location = evidence["location"]
    for suffix, value in (
        ("feature_start", location["feature_start"]),
        ("feature_end_exclusive", location["feature_end_exclusive"]),
        ("decision_support", evidence["decision_support"]),
        ("class_changed", evidence["class_changed"]),
        ("mapping_status", location["mapping_status"]),
        ("raw_text_fragment", location["raw_text_fragment"]),
        ("aligned_transcript_fragment", location["aligned_transcript_fragment"]),
        ("media_start_seconds", location["media_start_seconds"]),
        ("media_end_seconds", location["media_end_seconds"]),
        ("video_frame_index", location["video_frame_index"]),
        ("video_frame_time_seconds", location["video_frame_time_seconds"]),
    ):
        fields[f"{prefix}_{suffix}"] = "" if value is None else value
    return fields


def _check_record(record, expected_id):
    sample_id = record["sample_id"]
    if sample_id != expected_id or record["feature_version"] != "aligned":
        raise ValueError(f"Unexpected sample/version: {sample_id}")
    if not record["feature_file"].endswith(f"\\{sample_id}.pkl"):
        raise ValueError(f"Feature source ID mismatch: {sample_id}")
    if not record["video_file"].endswith(f"\\{sample_id}.mp4"):
        raise ValueError(f"Video source ID mismatch: {sample_id}")

    score = record["predicted_score"]
    lower = record["thresholds"]["negative_neutral"]
    upper = record["thresholds"]["neutral_positive"]
    expected_class = 0 if score < lower else 2 if score > upper else 1
    if (not math.isfinite(score) or not -3 <= score <= 3 or
            record["predicted_class"] != expected_class):
        raise ValueError(f"Score/class disagreement or invalid score: {sample_id}")

    if record["main_modality"] not in MODALITIES:
        raise ValueError(f"Invalid main modality: {sample_id}")
    primary = record["primary_evidence"]
    if primary["location"]["modality"] != record["main_modality"]:
        raise ValueError(f"Primary evidence/modality mismatch: {sample_id}")
    if primary["location"]["sample_id"] != sample_id:
        raise ValueError(f"Primary evidence/source mismatch: {sample_id}")

    for name in MODALITIES:
        for effect in ("modality_signed_support", "positive_support_share"):
            value = record[effect][name]
            if not math.isfinite(value):
                raise ValueError(f"Invalid {effect}/{name}: {sample_id}")
        windows = record["top_evidence"][name]
        for window in windows:
            location = window["location"]
            if location["sample_id"] != sample_id or location["modality"] != name:
                raise ValueError(f"Local evidence/source mismatch: {sample_id}")
            if not 0 <= location["feature_start"] < location["feature_end_exclusive"] <= 50:
                raise ValueError(f"Invalid evidence span: {sample_id}/{name}")
            status = location["mapping_status"]
            if status not in ("text_exact", "media_candidate", "feature_only"):
                raise ValueError(f"Invalid evidence status: {sample_id}/{name}")
            if status == "text_exact":
                start = location["raw_text_char_start"]
                end = location["raw_text_char_end_exclusive"]
                if record["raw_text"][start:end] != location["raw_text_fragment"]:
                    raise ValueError(f"Raw text span mismatch: {sample_id}")
            elif status == "feature_only":
                if (location["media_start_seconds"] is not None or
                        location["video_frame_index"] is not None):
                    raise ValueError(f"Unsupported media location: {sample_id}/{name}")
    if primary != record["top_evidence"][record["main_modality"]][0]:
        raise ValueError(f"Primary evidence is not the main modality's top window: {sample_id}")


def export():
    records = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if len(records) != 20:
        raise ValueError(f"Expected 20 samples, got {len(records)}")
    for index, record in enumerate(records, 1):
        _check_record(record, f"{index:02d}")

    rows = []
    for record in records:
        cls = record["predicted_class"]
        row = {
            "sample_id": record["sample_id"],
            "feature_version": record["feature_version"],
            "predicted_score": record["predicted_score"],
            "predicted_class": cls,
            "predicted_label": LABELS[cls],
            "main_modality": record["main_modality"],
            "main_supportive": record["main_supportive"],
            "main_uncertain": record["main_uncertain"],
            "fusion_main_modality": record["fusion_main_modality"],
        }
        for name in MODALITIES:
            row[f"shapley_decision_support_{name}"] = record["modality_signed_support"][name]
            row[f"positive_support_share_{name}"] = record["positive_support_share"][name]
        row.update(_location_fields("primary_evidence", record["primary_evidence"]))
        for name in MODALITIES:
            windows = record["top_evidence"][name]
            row.update(_location_fields(f"{name}_top_evidence", windows[0] if windows else None))
        row["warnings"] = " | ".join(record["warnings"])
        rows.append(row)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return OUTPUT


if __name__ == "__main__":
    print(export())
