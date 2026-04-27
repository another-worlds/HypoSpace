from __future__ import annotations

"""CLI entrypoint — parses flags and delegates to HypoSpaceAPI or run_diagnostics()."""

import argparse
import json

from api import HypoSpaceAPI
from core.config import DecoderConfig, GovernanceConfig, RuntimeConfig
from interpretability.faithfulness import GovernanceThresholdError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HypoSpace quickstart CLI")
    parser.add_argument("--model", default="demo-model", help="Model name")
    parser.add_argument("--layer", default="layer_0", help="Layer identifier")
    parser.add_argument("--activations", default="0.1,0.4,-0.2,0.8", help="Comma-separated activation values (raw path)")
    parser.add_argument("--top-k", type=int, default=8, help="How many strongest features to keep")
    parser.add_argument("--device", default="cpu", help="Runtime device")
    parser.add_argument("--min-faithfulness", type=float, default=0.65, help="Faithfulness threshold")
    parser.add_argument("--min-stability", type=float, default=0.60, help="Stability threshold")
    parser.add_argument("--fail-on-low-confidence", action="store_true", help="Fail command if governance thresholds are not met")
    parser.add_argument("--version", default="0.1.0", help="Kernel version tag")
    parser.add_argument("--diagnostics", action="store_true", help="Run deep diagnostics and print JSON report; ignores all other flags")
    parser.add_argument("--high-intensity-threshold", type=float, default=0.8, help="Score threshold for 'high-intensity' label")
    parser.add_argument("--medium-intensity-threshold", type=float, default=0.4, help="Score threshold for 'medium-intensity' label")
    # Live-model extraction flags (require nnsight + torch)
    parser.add_argument("--layer-path", default=None, help="Dot-notation path into model for live extraction (e.g. transformer.h.0); if set, --inputs is required and --activations is ignored")
    parser.add_argument("--inputs", default=None, help="Text input for live model extraction (used with --layer-path)")
    parser.add_argument("--token-index", type=int, default=-1, help="Token position to extract (-1 = last token)")
    return parser


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
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
