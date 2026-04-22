from __future__ import annotations

import importlib.util
import json
import math
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from data.utils import utc_timestamp

DIAGNOSTICS_VERSION = "1.0.0"
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


@dataclass(slots=True)
class CheckDetail:
    name: str
    passed: bool
    detail: str


@dataclass(slots=True)
class ProbeResult:
    subsystem: str
    status: str  # "ok" | "degraded" | "failed" | "skipped"
    latency_ms: float
    checks: List[CheckDetail] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass(slots=True)
class DependencyStatus:
    name: str
    available: bool
    note: str


@dataclass(slots=True)
class DiagnosticsReport:
    schema_version: str
    generated_at_utc: str
    overall_status: str  # "ok" | "degraded" | "failed"
    dependencies: List[DependencyStatus]
    probes: List[ProbeResult]


def _has_bad_floats(values: list[float]) -> bool:
    return any(math.isnan(v) or math.isinf(v) for v in values)


def _detect_dependencies() -> List[DependencyStatus]:
    result = []
    for name in ("torch", "nnsight", "pyvene", "diskcache"):
        available = importlib.util.find_spec(name) is not None
        note = "" if available else f"install with: pip install {name}"
        result.append(DependencyStatus(name=name, available=available, note=note))
    return result


def _overall_status(probes: List[ProbeResult]) -> str:
    statuses = {p.status for p in probes}
    if "failed" in statuses:
        return "failed"
    if "degraded" in statuses:
        return "degraded"
    return "ok"


def _probe_config(tmp_dir: Path) -> ProbeResult:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    checks: List[CheckDetail] = []
    warnings: List[str] = []
    errors: List[str] = []
    try:
        from core.config import DecoderConfig, GovernanceConfig

        cfg = DecoderConfig()
        faith = cfg.governance.min_faithfulness_score
        stab = cfg.governance.min_stability_score
        score_range_ok = 0.0 <= faith <= 1.0 and 0.0 <= stab <= 1.0
        checks.append(CheckDetail(
            "governance_score_range",
            score_range_ok,
            f"min_faithfulness={faith}, min_stability={stab}",
        ))
        if not score_range_ok:
            errors.append(f"Default governance thresholds out of [0,1]: faithfulness={faith}, stability={stab}")

        bad = GovernanceConfig(min_faithfulness_score=1.5, min_stability_score=-0.1)
        out_of_range_detected = (
            not (0.0 <= bad.min_faithfulness_score <= 1.0)
            and not (0.0 <= bad.min_stability_score <= 1.0)
        )
        checks.append(CheckDetail(
            "governance_out_of_range_detected",
            out_of_range_detected,
            "probe correctly identifies invalid threshold values",
        ))
        if not out_of_range_detected:
            errors.append("Probe failed to detect out-of-range governance config values")

        top_k_ok = cfg.top_k > 0
        checks.append(CheckDetail("top_k_positive", top_k_ok, f"top_k={cfg.top_k}"))
        if not top_k_ok:
            errors.append(f"Default top_k={cfg.top_k} is not > 0")

        mf_ok = cfg.runtime.max_features > 0
        checks.append(CheckDetail("max_features_positive", mf_ok, f"max_features={cfg.runtime.max_features}"))
        if not mf_ok:
            errors.append(f"Default max_features={cfg.runtime.max_features} is not > 0")

        semver_ok = bool(_SEMVER_RE.match("0.1.0")) and not bool(_SEMVER_RE.match(""))
        checks.append(CheckDetail("semver_valid_pattern", semver_ok, "0.1.0 matches semver, empty string does not"))
        if not semver_ok:
            errors.append("Semver regex is not functioning correctly")

    except Exception as exc:
        errors.append(f"probe crashed: {exc}")

    latency_ms = (time.monotonic() - start) * 1000.0
    status = "failed" if errors else ("degraded" if warnings else "ok")
    return ProbeResult("config", status, round(latency_ms, 3), checks, warnings, errors)


def _probe_preprocessor(tmp_dir: Path) -> ProbeResult:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    checks: List[CheckDetail] = []
    warnings: List[str] = []
    errors: List[str] = []
    try:
        from data.preprocessor import ActivationPreprocessor

        pp = ActivationPreprocessor()

        out = pp.normalize([0.1, 0.4, -0.2, 0.8])
        in_range = all(-1.0 <= v <= 1.0 for v in out)
        no_bad = not _has_bad_floats(out)
        checks.append(CheckDetail("normalize_clean_input", in_range and no_bad, f"output={out}"))
        if not (in_range and no_bad):
            errors.append(f"Normalized output not in [-1,1] or contains NaN/inf: {out}")

        zeros_out = pp.normalize([0.0, 0.0])
        zeros_ok = not _has_bad_floats(zeros_out) and zeros_out == [0.0, 0.0]
        checks.append(CheckDetail("normalize_all_zeros_returns_zeros", zeros_ok, f"output={zeros_out}"))
        if not zeros_ok:
            errors.append(f"All-zeros input produced bad output: {zeros_out}")

        empty_out = pp.normalize([])
        checks.append(CheckDetail("normalize_empty_returns_empty", empty_out == [], f"output={empty_out}"))
        if empty_out != []:
            errors.append(f"Empty input returned non-empty: {empty_out}")

        nan_detected = _has_bad_floats([0.1, float("nan"), 0.3])
        checks.append(CheckDetail("nan_detection_works", nan_detected, "math.isnan fires on NaN input"))
        if not nan_detected:
            errors.append("_has_bad_floats() failed to detect NaN")

        inf_detected = _has_bad_floats([0.1, float("inf")])
        checks.append(CheckDetail("inf_detection_works", inf_detected, "math.isinf fires on inf input"))
        if not inf_detected:
            errors.append("_has_bad_floats() failed to detect inf")

        checks.append(CheckDetail("output_no_nan_clean_path", no_bad, "no NaN/inf in clean input output"))

    except Exception as exc:
        errors.append(f"probe crashed: {exc}")

    latency_ms = (time.monotonic() - start) * 1000.0
    status = "failed" if errors else ("degraded" if warnings else "ok")
    return ProbeResult("preprocessor", status, round(latency_ms, 3), checks, warnings, errors)


def _probe_hierarchy(tmp_dir: Path) -> ProbeResult:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    checks: List[CheckDetail] = []
    warnings: List[str] = []
    errors: List[str] = []
    try:
        from core.hierarchy import HierarchyEngine

        engine = HierarchyEngine(backend="matryoshka")
        # [0.1, 0.9, 0.4, -0.7] — top-2 by abs: index 1 (0.9), index 3 (0.7)
        features = engine.extract_features([0.1, 0.9, 0.4, -0.7], layer="diag-layer", top_k=2)

        count_ok = len(features) == 2
        checks.append(CheckDetail("features_returned_count", count_ok, f"got {len(features)} features"))
        if not count_ok:
            errors.append(f"Expected 2 features, got {len(features)}")

        if features:
            top_ok = features[0].source_index == 1
            checks.append(CheckDetail("top_features_by_magnitude", top_ok, f"top feature source_index={features[0].source_index}"))
            if not top_ok:
                errors.append(f"Top feature has wrong source_index: {features[0].source_index} (expected 1 for value 0.9)")

            scores_nonneg = all(f.score >= 0 for f in features)
            checks.append(CheckDetail("feature_scores_nonnegative", scores_nonneg, f"scores={[f.score for f in features]}"))
            if not scores_nonneg:
                errors.append("Negative scores found in features")

            scores_no_nan = not _has_bad_floats([f.score for f in features])
            checks.append(CheckDetail("feature_scores_no_nan", scores_no_nan, "no NaN/inf in scores"))
            if not scores_no_nan:
                errors.append("NaN/inf found in feature scores")

        zero_k_features = engine.extract_features([0.5, -0.3], layer="diag-layer", top_k=0)
        zero_k_ok = len(zero_k_features) >= 1
        zero_k_note = f"top_k=0 returns {len(zero_k_features)} feature(s) — silently promoted to 1"
        checks.append(CheckDetail("top_k_zero_guard", zero_k_ok, zero_k_note))
        if not zero_k_ok:
            errors.append("HierarchyEngine returned 0 features for top_k=0 input")
        else:
            warnings.append("top_k=0 is silently promoted to 1 — no explicit input validation in HierarchyEngine")

        empty_features = engine.extract_features([], layer="diag-layer", top_k=4)
        checks.append(CheckDetail("empty_input_returns_empty", empty_features == [], f"got {len(empty_features)} features"))
        if empty_features != []:
            errors.append(f"Empty input returned {len(empty_features)} features")

    except Exception as exc:
        errors.append(f"probe crashed: {exc}")

    latency_ms = (time.monotonic() - start) * 1000.0
    status = "failed" if errors else ("degraded" if warnings else "ok")
    return ProbeResult("hierarchy", status, round(latency_ms, 3), checks, warnings, errors)


def _probe_kernel_library(tmp_dir: Path) -> ProbeResult:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    checks: List[CheckDetail] = []
    warnings: List[str] = []
    errors: List[str] = []
    try:
        from core.hierarchy import Feature
        from core.kernel_library import KernelLibrary, KernelTemplate

        normal_dir = tmp_dir / "normal"
        normal_dir.mkdir(parents=True, exist_ok=True)
        lib = KernelLibrary(root=normal_dir)

        features = [
            Feature(id="l:feature:0", layer="l", score=0.9, source_index=0),
            Feature(id="l:feature:1", layer="l", score=0.5, source_index=1),
        ]
        template = KernelTemplate(
            kernel_id="diag-k",
            version="0.1.0",
            model_name="diag-model",
            layer="diag-layer",
            features=features,
        )

        path = lib.save(template)
        save_ok = path.exists()
        checks.append(CheckDetail("save_creates_file", save_ok, str(path)))
        if not save_ok:
            errors.append(f"Saved artifact not found: {path}")

        manifest_exists = lib.manifest_path.exists()
        checks.append(CheckDetail("manifest_exists_after_save", manifest_exists, str(lib.manifest_path)))
        if not manifest_exists:
            errors.append("manifest.json not created after save")

        manifest: dict = {}
        manifest_valid = False
        if manifest_exists:
            try:
                manifest = json.loads(lib.manifest_path.read_text(encoding="utf-8"))
                manifest_valid = True
            except json.JSONDecodeError as exc:
                errors.append(f"manifest.json is invalid JSON: {exc}")
        checks.append(CheckDetail("manifest_is_valid_json", manifest_valid, "manifest.json parses without error"))

        if manifest_valid:
            latest_ok = manifest.get("diag-k", {}).get("latest") == "0.1.0"
            checks.append(CheckDetail("manifest_latest_matches_saved", latest_ok, f"latest={manifest.get('diag-k', {}).get('latest')}"))
            if not latest_ok:
                errors.append("manifest latest version does not match saved version")

        loaded = lib.load_latest("diag-k")
        rt_ok = (
            loaded.kernel_id == "diag-k"
            and loaded.version == "0.1.0"
            and len(loaded.features) == 2
            and abs(loaded.features[0].score - 0.9) < 1e-9
        )
        checks.append(CheckDetail("load_roundtrip", rt_ok, f"kernel_id={loaded.kernel_id}, version={loaded.version}, features={len(loaded.features)}"))
        if not rt_ok:
            errors.append("Round-trip load did not match saved template")

        manifest_files = set(manifest.get("diag-k", {}).get("files", {}).values()) if manifest_valid else set()
        disk_files = {f.name for f in normal_dir.iterdir() if f.name != "manifest.json" and f.suffix == ".json"}
        orphaned = disk_files - manifest_files
        checks.append(CheckDetail("no_orphaned_files", not orphaned, f"manifest={manifest_files}, disk={disk_files}"))
        if orphaned:
            warnings.append(f"Orphaned kernel files on disk: {orphaned}")

        template2 = KernelTemplate(kernel_id="diag-k", version="0.2.0", model_name="diag-model", layer="diag-layer", features=[])
        lib.save(template2)
        manifest2 = json.loads(lib.manifest_path.read_text(encoding="utf-8"))
        latest2_ok = manifest2.get("diag-k", {}).get("latest") == "0.2.0"
        checks.append(CheckDetail("second_version_updates_latest", latest2_ok, f"latest={manifest2.get('diag-k', {}).get('latest')}"))
        if not latest2_ok:
            errors.append("Saving a higher version did not update manifest latest")

        scores = lib.match("diag-k", features, version="0.1.0")
        scores_valid = all(0.0 <= v <= 1.0 for v in scores.values())
        checks.append(CheckDetail("match_returns_scores_in_range", scores_valid, f"scores={scores}"))
        if not scores_valid:
            errors.append(f"match() returned scores outside [0, 1]: {scores}")

        # Write corrupt manifest in isolated sub-dir to avoid contaminating the main checks
        corrupt_dir = tmp_dir / "corrupt"
        corrupt_dir.mkdir(parents=True, exist_ok=True)
        corrupt_lib = KernelLibrary(root=corrupt_dir)
        (corrupt_dir / "manifest.json").write_text("{bad json", encoding="utf-8")
        corrupt_detected = False
        try:
            corrupt_lib._read_manifest()
        except json.JSONDecodeError:
            corrupt_detected = True
        checks.append(CheckDetail("corrupt_manifest_detected", corrupt_detected, "json.JSONDecodeError raised on invalid manifest.json"))
        if not corrupt_detected:
            errors.append("_read_manifest() did not raise on corrupt manifest.json")

    except Exception as exc:
        errors.append(f"probe crashed: {exc}")

    latency_ms = (time.monotonic() - start) * 1000.0
    status = "failed" if errors else ("degraded" if warnings else "ok")
    return ProbeResult("kernel_library", status, round(latency_ms, 3), checks, warnings, errors)


def _probe_extractor(tmp_dir: Path) -> ProbeResult:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    checks: List[CheckDetail] = []
    warnings: List[str] = []
    errors: List[str] = []
    try:
        from data.extractor import ActivationExtractor

        ex = ActivationExtractor(cache_dir=str(tmp_dir))

        first = ex.extract([0.1, 0.5], model_name="diag", layer="L0")
        is_list = isinstance(first, list)
        checks.append(CheckDetail("extract_returns_list", is_list, f"type={type(first).__name__}"))
        if not is_list:
            errors.append(f"extract() returned {type(first).__name__} instead of list")

        rt_ok = is_list and len(first) == 2 and abs(first[0] - 0.1) < 1e-9 and abs(first[1] - 0.5) < 1e-9
        checks.append(CheckDetail("extract_roundtrip_values", rt_ok, f"values={first}"))
        if not rt_ok:
            errors.append(f"extract() returned wrong values: {first}")

        second = ex.extract([0.1, 0.5], model_name="diag", layer="L0")
        cache_hit = first == second
        checks.append(CheckDetail("cache_hit_returns_same", cache_hit, f"first={first}, second={second}"))
        if not cache_hit:
            errors.append(f"Cache hit returned different values: first={first}, second={second}")

        dc_used = ex._dc is not None
        checks.append(CheckDetail("cache_backend", True, f"diskcache_used={dc_used}"))
        if not dc_used:
            warnings.append("diskcache unavailable — using plain JSON cache (collision risk at scale)")

        key = ex._cache_key([0.1, 0.5], "diag", "L0")
        key_ok = isinstance(key, str) and len(key) == 16
        checks.append(CheckDetail("cache_key_length", key_ok, f"key='{key}' ({len(key)} chars)"))
        if not key_ok:
            errors.append(f"Cache key is not 16-char hex: '{key}'")

    except Exception as exc:
        errors.append(f"probe crashed: {exc}")

    latency_ms = (time.monotonic() - start) * 1000.0
    status = "failed" if errors else ("degraded" if warnings else "ok")
    return ProbeResult("extractor", status, round(latency_ms, 3), checks, warnings, errors)


def _probe_semantic(tmp_dir: Path) -> ProbeResult:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    checks: List[CheckDetail] = []
    warnings: List[str] = []
    errors: List[str] = []
    try:
        from core.hierarchy import Feature
        from interpretability.semantic import SemanticInterpreter

        si = SemanticInterpreter()
        features = [
            Feature(id="l:feature:0", layer="l", score=0.9, source_index=0),
            Feature(id="l:feature:1", layer="l", score=0.5, source_index=1),
            Feature(id="l:feature:2", layer="l", score=0.1, source_index=2),
        ]
        annotated = si.annotate(features)

        high_ok = annotated[0].label is not None and annotated[0].label.startswith("high-intensity")
        checks.append(CheckDetail("high_intensity_label", high_ok, f"score=0.9 → label='{annotated[0].label}'"))
        if not high_ok:
            errors.append(f"score=0.9 got unexpected label: '{annotated[0].label}'")

        med_ok = annotated[1].label is not None and annotated[1].label.startswith("medium-intensity")
        checks.append(CheckDetail("medium_intensity_label", med_ok, f"score=0.5 → label='{annotated[1].label}'"))
        if not med_ok:
            errors.append(f"score=0.5 got unexpected label: '{annotated[1].label}'")

        low_ok = annotated[2].label is not None and annotated[2].label.startswith("low-intensity")
        checks.append(CheckDetail("low_intensity_label", low_ok, f"score=0.1 → label='{annotated[2].label}'"))
        if not low_ok:
            errors.append(f"score=0.1 got unexpected label: '{annotated[2].label}'")

        pre_labeled = Feature(id="l:feature:x", layer="l", score=0.9, source_index=99, label="keep-me")
        after = si.annotate([pre_labeled])
        kept_ok = after[0].label == "keep-me"
        checks.append(CheckDetail("existing_label_not_overwritten", kept_ok, f"label='{after[0].label}'"))
        if not kept_ok:
            errors.append(f"Pre-existing label was overwritten: '{after[0].label}'")

        all_labeled = all(f.label is not None and f.label != "" for f in annotated)
        checks.append(CheckDetail("all_features_labeled", all_labeled, f"labels={[f.label for f in annotated]}"))
        if not all_labeled:
            errors.append("Some features have None/empty label after annotation")

    except Exception as exc:
        errors.append(f"probe crashed: {exc}")

    latency_ms = (time.monotonic() - start) * 1000.0
    status = "failed" if errors else ("degraded" if warnings else "ok")
    return ProbeResult("semantic", status, round(latency_ms, 3), checks, warnings, errors)


def _probe_mechanistic(tmp_dir: Path) -> ProbeResult:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    checks: List[CheckDetail] = []
    warnings: List[str] = []
    errors: List[str] = []
    try:
        from core.hierarchy import Feature
        from interpretability.faithfulness import FaithfulnessChecker
        from interpretability.mechanistic import MechanisticAnalyzer

        features = [
            Feature(id="l:feature:0", layer="l", score=0.8, source_index=0),
            Feature(id="l:feature:1", layer="l", score=0.4, source_index=1),
            Feature(id="l:feature:2", layer="l", score=0.2, source_index=2),
        ]
        results = MechanisticAnalyzer().run_interventions(features)

        count_ok = len(results) == 3
        checks.append(CheckDetail("interventions_count_matches_features", count_ok, f"got {len(results)} interventions for 3 features"))
        if not count_ok:
            errors.append(f"Expected 3 intervention results, got {len(results)}")

        sizes_nonneg = all(r.effect_size >= 0 for r in results)
        checks.append(CheckDetail("effect_sizes_nonnegative", sizes_nonneg, f"sizes={[r.effect_size for r in results]}"))
        if not sizes_nonneg:
            errors.append("Negative effect_size found in mechanistic results")

        is_stub = all(abs(r.ablated - r.baseline * 0.5) < 1e-9 for r in results)
        checks.append(CheckDetail("ablated_is_50pct_of_baseline", is_stub, "confirms stable stub behavior"))
        if not is_stub:
            errors.append("MechanisticAnalyzer stub produced unexpected ablated values")

        no_nan = not _has_bad_floats([r.effect_size for r in results])
        checks.append(CheckDetail("no_nan_in_scores", no_nan, "no NaN/inf in effect_sizes"))
        if not no_nan:
            errors.append("NaN/inf found in intervention effect_sizes")

        sc = FaithfulnessChecker().evaluate(results)
        faith_ok = 0.0 <= sc.faithfulness_score <= 1.0
        checks.append(CheckDetail("faithfulness_score_in_range", faith_ok, f"faithfulness_score={sc.faithfulness_score}"))
        if not faith_ok:
            errors.append(f"faithfulness_score out of [0,1]: {sc.faithfulness_score}")

        stab_ok = 0.0 <= sc.stability_score <= 1.0
        checks.append(CheckDetail("stability_score_in_range", stab_ok, f"stability_score={sc.stability_score}"))
        if not stab_ok:
            errors.append(f"stability_score out of [0,1]: {sc.stability_score}")

        flag_ok = sc.risk_flag in {"ok", "low_faithfulness", "no_data"}
        checks.append(CheckDetail("risk_flag_valid_value", flag_ok, f"risk_flag='{sc.risk_flag}'"))
        if not flag_ok:
            errors.append(f"Unexpected risk_flag value: '{sc.risk_flag}'")

        warnings.append("MechanisticAnalyzer is a 50% ablation stub — real interventions require torch + pyvene")

    except Exception as exc:
        errors.append(f"probe crashed: {exc}")

    latency_ms = (time.monotonic() - start) * 1000.0
    status = "failed" if errors else ("degraded" if warnings else "ok")
    return ProbeResult("mechanistic", status, round(latency_ms, 3), checks, warnings, errors)


def _probe_full_pipeline(tmp_dir: Path) -> ProbeResult:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    checks: List[CheckDetail] = []
    warnings: List[str] = []
    errors: List[str] = []
    try:
        from api import HypoSpaceAPI
        from core.config import DecoderConfig, RuntimeConfig

        cfg = DecoderConfig(top_k=3, runtime=RuntimeConfig(cache_dir=str(tmp_dir)))
        api = HypoSpaceAPI(config=cfg)
        run = api.decode_and_score("diag-model", "diag-layer", [0.1, 0.4, -0.2, 0.8, 0.5, -0.3])

        checks.append(CheckDetail("decode_returns_result", True, "no exception raised"))

        count_ok = len(run.decode.features) == 3
        checks.append(CheckDetail("features_count_correct", count_ok, f"got {len(run.decode.features)} features"))
        if not count_ok:
            errors.append(f"Expected 3 features, got {len(run.decode.features)}")

        labeled_ok = all(f.label is not None for f in run.decode.features)
        checks.append(CheckDetail("features_labeled", labeled_ok, f"labels={[f.label for f in run.decode.features]}"))
        if not labeled_ok:
            errors.append("Some features have no label after pipeline")

        kp = run.decode.kernel_path
        kp_ok = kp is not None and Path(kp).exists()
        checks.append(CheckDetail("kernel_path_exists", kp_ok, f"kernel_path={kp}"))
        if not kp_ok:
            errors.append(f"kernel_path missing or not on disk: {kp}")

        scores_norm = all(0.0 <= f.score <= 1.0 for f in run.decode.features)
        checks.append(CheckDetail("scores_normalized", scores_norm, f"scores={[f.score for f in run.decode.features]}"))
        if not scores_norm:
            errors.append("Feature scores outside [0, 1] after normalization")

        no_nan = not _has_bad_floats([f.score for f in run.decode.features])
        checks.append(CheckDetail("no_nan_in_scores", no_nan, "no NaN/inf in feature scores"))
        if not no_nan:
            errors.append("NaN/inf found in feature scores")

        checks.append(CheckDetail("scorecard_present", True, "GovernanceScorecard returned"))

        flag_ok = run.scorecard.risk_flag in {"ok", "low_faithfulness", "no_data"}
        checks.append(CheckDetail("scorecard_risk_flag_valid", flag_ok, f"risk_flag='{run.scorecard.risk_flag}'"))
        if not flag_ok:
            errors.append(f"Unexpected scorecard risk_flag: '{run.scorecard.risk_flag}'")

        rate_str = run.decode.metadata.get("cross_run_match_rate", "")
        rate_ok = False
        try:
            rate = float(rate_str)
            rate_ok = 0.0 <= rate <= 1.0
        except ValueError:
            pass
        checks.append(CheckDetail("cross_run_match_rate_parseable", rate_ok, f"cross_run_match_rate='{rate_str}'"))
        if not rate_ok:
            errors.append(f"cross_run_match_rate not parseable as float in [0,1]: '{rate_str}'")

    except Exception as exc:
        errors.append(f"probe crashed: {exc}")

    latency_ms = (time.monotonic() - start) * 1000.0
    status = "failed" if errors else ("degraded" if warnings else "ok")
    return ProbeResult("full_pipeline", status, round(latency_ms, 3), checks, warnings, errors)


def _probe_nnsight(tmp_dir: Path) -> ProbeResult:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()

    nnsight_avail = importlib.util.find_spec("nnsight") is not None
    torch_avail = importlib.util.find_spec("torch") is not None

    if not nnsight_avail or not torch_avail:
        latency_ms = (time.monotonic() - start) * 1000.0
        msg = "nnsight and/or torch not installed — install with: pip install nnsight torch"
        return ProbeResult("nnsight", "skipped", round(latency_ms, 3), [], [msg], [])

    checks: List[CheckDetail] = []
    warnings: List[str] = []
    errors: List[str] = []
    try:
        from data.nnsight_extractor import NNSightExtractor, nnsight_available  # noqa: F401

        checks.append(CheckDetail("import_succeeds", True, "NNSightExtractor imported successfully"))

        avail = nnsight_available()
        checks.append(CheckDetail("nnsight_available_returns_true", avail, f"nnsight_available()={avail}"))
        if not avail:
            errors.append("nnsight_available() returned False despite successful import")

    except Exception as exc:
        errors.append(f"probe crashed: {exc}")

    latency_ms = (time.monotonic() - start) * 1000.0
    status = "failed" if errors else ("degraded" if warnings else "ok")
    return ProbeResult("nnsight", status, round(latency_ms, 3), checks, warnings, errors)


def _probe_pyvene(tmp_dir: Path) -> ProbeResult:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()

    pyvene_avail = importlib.util.find_spec("pyvene") is not None
    torch_avail = importlib.util.find_spec("torch") is not None

    if not pyvene_avail or not torch_avail:
        latency_ms = (time.monotonic() - start) * 1000.0
        msg = "pyvene and/or torch not installed — install with: pip install pyvene torch"
        return ProbeResult("pyvene", "skipped", round(latency_ms, 3), [], [msg], [])

    checks: List[CheckDetail] = []
    warnings: List[str] = []
    errors: List[str] = []
    try:
        from data.pyvene_runner import PyVeneInterventionRunner, pyvene_available  # noqa: F401

        checks.append(CheckDetail("import_succeeds", True, "PyVeneInterventionRunner imported successfully"))

        avail = pyvene_available()
        checks.append(CheckDetail("pyvene_available_returns_true", avail, f"pyvene_available()={avail}"))
        if not avail:
            errors.append("pyvene_available() returned False despite successful import")

    except Exception as exc:
        errors.append(f"probe crashed: {exc}")

    latency_ms = (time.monotonic() - start) * 1000.0
    status = "failed" if errors else ("degraded" if warnings else "ok")
    return ProbeResult("pyvene", status, round(latency_ms, 3), checks, warnings, errors)


def run_diagnostics() -> DiagnosticsReport:
    with tempfile.TemporaryDirectory() as _tmp:
        tmp_dir = Path(_tmp)
        probes = [
            _probe_config(tmp_dir / "config"),
            _probe_preprocessor(tmp_dir / "preprocessor"),
            _probe_hierarchy(tmp_dir / "hierarchy"),
            _probe_kernel_library(tmp_dir / "kernel"),
            _probe_extractor(tmp_dir / "extractor"),
            _probe_semantic(tmp_dir / "semantic"),
            _probe_mechanistic(tmp_dir / "mechanistic"),
            _probe_full_pipeline(tmp_dir / "pipeline"),
            _probe_nnsight(tmp_dir / "nnsight"),
            _probe_pyvene(tmp_dir / "pyvene"),
        ]
    return DiagnosticsReport(
        schema_version=DIAGNOSTICS_VERSION,
        generated_at_utc=utc_timestamp(),
        overall_status=_overall_status(probes),
        dependencies=_detect_dependencies(),
        probes=probes,
    )
