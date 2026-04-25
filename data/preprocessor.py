from __future__ import annotations

from typing import Iterable, List


class ActivationPreprocessor:
    """Normalize activation vectors for stable downstream decoding."""

    def normalize(self, values: Iterable[float]) -> List[float]:
        numbers = [float(v) for v in values]
        if not numbers:
            return []
        # ISSUE-H02: NaN/inf in input → max_abs = inf; outputs become 0 or NaN without any error raised
        max_abs = max(abs(v) for v in numbers) or 1.0
        return [v / max_abs for v in numbers]
