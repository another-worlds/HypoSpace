from __future__ import annotations

"""CLI entrypoint — parses flags and delegates to HypoSpaceAPI or run_diagnostics()."""

import argparse
import csv
import json
from pathlib import Path

from api import HypoSpaceAPI
from core.config import DecoderConfig, GovernanceConfig, RuntimeConfig
from interpretability.faithfulness import GovernanceThresholdError


def _float_in_unit_interval(value: str) -> float:
    v = float(value)
    if not (0.0 <= v <= 1.0):
        raise argparse.ArgumentTypeError(f"must be in [0, 1], got {v}")
    return v


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HypoSpace quickstart CLI")
    parser.add_argument("--model", default="demo-model", help="Model name")
    parser.add_argument("--layer", default="layer_0", help="Layer identifier")
    parser.add_argument("--activations", default="0.1,0.4,-0.2,0.8", help="Comma-separated activation values (raw path)")
    parser.add_argument("--top-k", type=int, default=8, help="How many strongest features to keep")
    parser.add_argument("--device", default="cpu", help="Runtime device")
    parser.add_argument("--min-faithfulness", type=_float_in_unit_interval, default=0.65, help="Faithfulness threshold")
    parser.add_argument("--min-stability", type=_float_in_unit_interval, default=0.60, help="Stability threshold")
    parser.add_argument("--fail-on-low-confidence", action="store_true", help="Fail command if governance thresholds are not met")
    parser.add_argument("--version", default="0.1.0", help="Kernel version tag")
    parser.add_argument("--diagnostics", action="store_true", help="Run deep diagnostics and print JSON report; ignores all other flags")
    parser.add_argument("--high-intensity-threshold", type=_float_in_unit_interval, default=0.8, help="Score threshold for 'high-intensity' label")
    parser.add_argument("--medium-intensity-threshold", type=_float_in_unit_interval, default=0.4, help="Score threshold for 'medium-intensity' label")
    # Live-model extraction flags (require nnsight + torch)
    parser.add_argument("--layer-path", default=None, help="Dot-notation path into model for live extraction (e.g. transformer.h.0); if set, --inputs is required and --activations is ignored")
    parser.add_argument("--inputs", default=None, help="Text input for live model extraction (used with --layer-path)")
    parser.add_argument("--token-index", type=int, default=-1, help="Token position to extract (-1 = last token)")
    # Batch flags
    parser.add_argument("--input-file", default=None, help="CSV file with one activation vector per row; when set, --activations is ignored and output is NDJSON")
    parser.add_argument("--output-file", default=None, help="Write output to this file instead of stdout")
    return parser


def _read_activation_csv(path: Path) -> list[list[float]]:
    """Parse a CSV file where each row is a comma-separated activation vector.

    A header row (where the first cell is not parseable as float) is silently
    skipped. Raises SystemExit(1) on missing file or non-numeric cell in data rows.
    """
    if not path.exists():
        print(json.dumps({"error": f"Input file not found: {path}", "type": "FileNotFoundError"}, ensure_ascii=False))
        raise SystemExit(1)
    rows: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for line_no, raw_row in enumerate(reader, start=1):
            if not raw_row:
                continue
            first_cell = raw_row[0].strip()
            if line_no == 1 and first_cell:
                try:
                    float(first_cell)
                except ValueError:
                    continue  # header row — skip
            try:
                parsed = [float(cell.strip()) for cell in raw_row if cell.strip()]
            except ValueError as exc:
                print(json.dumps({"error": f"Non-numeric value on line {line_no}: {exc}", "type": "ValueError"}, ensure_ascii=False))
                raise SystemExit(1) from exc
            if parsed:
                rows.append(parsed)
    return rows


def _write_output(text: str, output_file: str | None) -> None:
    if output_file is None:
        print(text, end="" if text.endswith("\n") else "\n")
    else:
        content = text if text.endswith("\n") else text + "\n"
        Path(output_file).write_text(content, encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.diagnostics:
        import dataclasses
        from diagnostics import run_diagnostics
        report = run_diagnostics()
        print(json.dumps(dataclasses.asdict(report), ensure_ascii=False, indent=2))
        raise SystemExit(0)

    config = DecoderConfig(
        top_k=args.top_k,
        high_intensity_threshold=args.high_intensity_threshold,
        medium_intensity_threshold=args.medium_intensity_threshold,
        runtime=RuntimeConfig(device=args.device),
        governance=GovernanceConfig(
            min_faithfulness_score=args.min_faithfulness,
            min_stability_score=args.min_stability,
            fail_on_low_confidence=args.fail_on_low_confidence,
        ),
    )
    api = HypoSpaceAPI(config=config)

    # ------------------------------------------------------------------
    # Batch mode: --input-file overrides --activations and outputs NDJSON
    # ------------------------------------------------------------------
    if args.input_file is not None:
        matrix = _read_activation_csv(Path(args.input_file))
        try:
            runs = api.decode_and_score_batch(
                model_name=args.model,
                layer=args.layer,
                activation_matrix=matrix,
                version=args.version,
            )
        except GovernanceThresholdError as exc:
            print(json.dumps({"error": str(exc), "type": "GovernanceThresholdError"}, ensure_ascii=False))
            raise SystemExit(2) from exc
        lines = [json.dumps(r.to_dict(), ensure_ascii=False) for r in runs]
        _write_output("\n".join(lines) + "\n", args.output_file)
        return

    # ------------------------------------------------------------------
    # Single-decode mode
    # ------------------------------------------------------------------
    try:
        if args.layer_path is not None:
            if args.inputs is None:
                print(json.dumps({"error": "--inputs is required when --layer-path is set", "type": "ArgumentError"}, ensure_ascii=False, indent=2))
                raise SystemExit(1)
            try:
                run = api.decode_and_score_from_model(
                    model_name=args.model,
                    layer=args.layer,
                    layer_path=args.layer_path,
                    inputs=args.inputs,
                    token_index=args.token_index,
                    version=args.version,
                )
            except ImportError as exc:
                print(json.dumps({"error": f"Live model extraction requires nnsight and torch: {exc}", "type": "ImportError"}, ensure_ascii=False, indent=2))
                raise SystemExit(1) from exc
        else:
            raw_activations = [float(v.strip()) for v in args.activations.split(",") if v.strip()]
            run = api.decode_and_score(
                model_name=args.model,
                layer=args.layer,
                raw_activations=raw_activations,
                version=args.version,
            )
    except GovernanceThresholdError as exc:
        print(json.dumps({"error": str(exc), "type": "GovernanceThresholdError"}, ensure_ascii=False, indent=2))
        raise SystemExit(2) from exc

    payload = {
        "model": run.decode.model_name,
        "layer": run.decode.layer,
        "kernel_path": run.decode.kernel_path,
        "features": [
            {
                "id": f.id,
                "source_index": f.source_index,
                "score": f.score,
                "label": f.label,
            }
            for f in run.decode.features
        ],
        "scorecard": {
            "faithfulness_score": run.scorecard.faithfulness_score,
            "stability_score": run.scorecard.stability_score,
            "risk_flag": run.scorecard.risk_flag,
            "passes_thresholds": run.scorecard.passes_thresholds,
        },
    }
    output = json.dumps(payload, ensure_ascii=False, indent=2)
    _write_output(output, args.output_file)


if __name__ == "__main__":
    main()
