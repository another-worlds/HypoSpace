from __future__ import annotations

import argparse
import json

from api import HypoSpaceAPI
from core.config import DecoderConfig, GovernanceConfig, RuntimeConfig
from interpretability.faithfulness import GovernanceThresholdError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HypoSpace quickstart CLI")
    parser.add_argument("--model", default="demo-model", help="Model name")
    parser.add_argument("--layer", default="layer_0", help="Layer identifier")
    parser.add_argument("--activations", default="0.1,0.4,-0.2,0.8", help="Comma-separated activation values")
    parser.add_argument("--top-k", type=int, default=8, help="How many strongest features to keep")
    parser.add_argument("--device", default="cpu", help="Runtime device")
    parser.add_argument("--min-faithfulness", type=float, default=0.65, help="Faithfulness threshold")
    parser.add_argument("--min-stability", type=float, default=0.60, help="Stability threshold")
    parser.add_argument("--fail-on-low-confidence", action="store_true", help="Fail command if governance thresholds are not met")
    parser.add_argument("--version", default="0.1.0", help="Kernel version tag")
    parser.add_argument("--diagnostics", action="store_true", help="Run deep diagnostics and print JSON report; ignores all other flags")
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

    raw_activations = [float(v.strip()) for v in args.activations.split(",") if v.strip()]

    config = DecoderConfig(
        top_k=args.top_k,
        runtime=RuntimeConfig(device=args.device),
        governance=GovernanceConfig(
            min_faithfulness_score=args.min_faithfulness,
            min_stability_score=args.min_stability,
            fail_on_low_confidence=args.fail_on_low_confidence,
        ),
    )
    api = HypoSpaceAPI(config=config)

    try:
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
