from typing import Dict, List, Tuple


def reciprocal_rank_fusion(
    ranked_lists: List[List[int]], k: int = 60) -> List[Tuple[int, float]]:
    scores: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
