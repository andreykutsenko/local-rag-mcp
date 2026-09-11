"""Reciprocal Rank Fusion. Knows nothing about search: takes ranked id lists, returns fused scores."""

DEFAULT_K = 60


def reciprocal_rank_fusion(runs, k=DEFAULT_K, weights=None):
    """Fuse ranked lists of chunk ids into [(chunk_id, score)], best first.

    runs: list of ranked lists; position 0 is rank 1. score(d) = sum(w_i / (k + rank_i(d)))
    over the runs that contain d; a document found by one run gets one term.
    weights: optional per-run multipliers (default 1.0 each). Ties keep the order
    of first appearance across runs, so the result is deterministic.
    """
    if weights is None:
        weights = [1.0] * len(runs)
    if len(weights) != len(runs):
        raise ValueError(f"{len(weights)} weights for {len(runs)} runs")

    scores = {}
    for run, weight in zip(runs, weights):
        for position, chunk_id in enumerate(run):
            rank = position + 1
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)

    order = {chunk_id: index for index, chunk_id in reversed(list(enumerate(scores)))}
    return sorted(scores.items(), key=lambda item: (-item[1], order[item[0]]))
