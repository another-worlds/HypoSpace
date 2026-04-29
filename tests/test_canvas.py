from __future__ import annotations

from core.hierarchy import Feature
from viz.canvas import SemanticCanvas


def _feat(i: int, score: float) -> Feature:
    return Feature(id=f"f{i}", layer="l", score=score, source_index=i, label="lbl")


def test_single_feature_point() -> None:
    pts = SemanticCanvas().to_points([_feat(0, 0.9)])
    assert len(pts) == 1


def test_zero_score_feature() -> None:
    pts = SemanticCanvas().to_points([_feat(0, 0.0)])
    assert pts[0][1] == 0.0  # (id, score) tuple


def test_no_features_returns_empty() -> None:
    assert SemanticCanvas().to_points([]) == []
    assert SemanticCanvas().to_edges([]) == []


def test_single_feature_no_edges() -> None:
    assert SemanticCanvas().to_edges([_feat(0, 0.5)]) == []


def test_multiple_features_have_edges() -> None:
    features = [_feat(i, 0.1 * (i + 1)) for i in range(5)]
    edges = SemanticCanvas().to_edges(features)
    assert len(edges) > 0


def test_points_sorted_by_score_descending() -> None:
    features = [_feat(0, 0.3), _feat(1, 0.9), _feat(2, 0.5)]
    pts = SemanticCanvas().to_points(features)
    scores = [score for _, score in pts]
    assert scores == sorted(scores, reverse=True)


def test_edges_connect_adjacent_sorted_features() -> None:
    features = [_feat(0, 0.8), _feat(1, 0.6), _feat(2, 0.4)]
    edges = SemanticCanvas().to_edges(features)
    assert len(edges) == 2
    assert edges[0] == ("f0", "f1", 0.7)
    assert edges[1] == ("f1", "f2", 0.5)


# ---------------------------------------------------------------------------
# SemanticCanvas.to_layout()
# ---------------------------------------------------------------------------

def test_to_layout_empty_returns_empty() -> None:
    assert SemanticCanvas().to_layout([]) == []


def test_to_layout_single_feature_returns_half_coords() -> None:
    layout = SemanticCanvas().to_layout([_feat(0, 0.9)])
    assert len(layout) == 1
    assert layout[0]["x"] == 0.5
    assert layout[0]["y"] == 0.5


def test_to_layout_xy_in_unit_interval() -> None:
    features = [_feat(i, 0.1 * (i + 1)) for i in range(5)]
    layout = SemanticCanvas().to_layout(features)
    assert len(layout) == 5
    for point in layout:
        assert 0.0 <= point["x"] <= 1.0
        assert 0.0 <= point["y"] <= 1.0


def test_to_layout_length_matches_features() -> None:
    features = [_feat(i, float(i)) for i in range(4)]
    layout = SemanticCanvas().to_layout(features)
    assert len(layout) == len(features)


def test_to_layout_highest_score_has_x_one() -> None:
    features = [_feat(0, 0.9), _feat(1, 0.5), _feat(2, 0.1)]
    layout = SemanticCanvas().to_layout(features)
    # after sort descending by score, rank 0 (highest) gets x=1.0
    top = next(p for p in layout if p["id"] == "f0")
    assert top["x"] == 1.0
