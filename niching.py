"""
Dynamic niching helpers for CVNAR/CVOA.

Structural distance ignores numeric intervals and uses Jaccard on
(attribute_index, role) pairs where role is antecedent (1) or consequent (2).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence


def active_attr_roles(attribute_type: Sequence[int]) -> frozenset[tuple[int, int]]:
    """Return set of (attr_idx, role) for attributes used in the rule."""
    roles = set()
    n = len(attribute_type) // 2
    for i in range(n):
        role = int(attribute_type[i * 2])
        if role in (1, 2):
            roles.add((i, role))
    return frozenset(roles)


def structure_fingerprint(attribute_type: Sequence[int]) -> tuple[int, ...]:
    """Exact structural signature (ant/cons/none per attribute)."""
    return tuple(int(attribute_type[i]) for i in range(0, len(attribute_type), 2))


def jaccard_similarity(set_a: frozenset, set_b: frozenset) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def structural_distance(attr_a: Sequence[int], attr_b: Sequence[int]) -> float:
    """Jaccard distance in [0, 1]: 0 = identical structure, 1 = disjoint."""
    return 1.0 - jaccard_similarity(active_attr_roles(attr_a), active_attr_roles(attr_b))


def sharing_function(distance: float, sharing_radius: float, alpha: float = 1.0) -> float:
    if sharing_radius <= 0:
        return 1.0 if distance == 0.0 else 0.0
    if distance < sharing_radius:
        return 1.0 - (distance / sharing_radius) ** alpha
    return 0.0


def niche_count(
    individual,
    population: Iterable,
    sharing_radius: float,
    alpha: float = 1.0,
) -> float:
    """Sum of sharing contributions from population (includes self => >= 1)."""
    attr_i = individual.attributeType
    count = 0.0
    for other in population:
        d = structural_distance(attr_i, other.attributeType)
        count += sharing_function(d, sharing_radius, alpha)
    return max(count, 1.0)


def shared_fitness(raw_fitness: float, niche_cnt: float) -> float:
    if niche_cnt <= 0:
        return raw_fitness
    return raw_fitness / niche_cnt


def apply_fitness_sharing(
    population: list,
    sharing_radius: float,
    alpha: float = 1.0,
) -> None:
    """
    Set individual.raw_fitness and individual.shared_fitness / fitness.

    Requires individual.fitness to already hold the raw objective value.
    After this call, individual.fitness becomes the shared fitness used for
    selection/propagation ranking.
    """
    if not population:
        return
    for ind in population:
        raw = ind.fitness
        if raw is None:
            continue
        ind.raw_fitness = raw
        m = niche_count(ind, population, sharing_radius, alpha)
        ind.niche_count = m
        ind.shared_fitness = shared_fitness(float(raw), m)
        ind.fitness = ind.shared_fitness


def count_structure_in_archive(archive: list, fingerprint: tuple[int, ...]) -> int:
    return sum(1 for ind in archive if structure_fingerprint(ind.attributeType) == fingerprint)


def worst_in_structure(archive: list, fingerprint: tuple[int, ...]):
    """Return (index, individual) of worst raw/obj fitness in a structural niche."""
    worst_idx = None
    worst_ind = None
    worst_fit = None
    for i, ind in enumerate(archive):
        if structure_fingerprint(ind.attributeType) != fingerprint:
            continue
        fit = getattr(ind, "raw_fitness", None)
        if fit is None:
            fit = ind.fitness
        if worst_fit is None or fit < worst_fit:
            worst_fit = fit
            worst_idx = i
            worst_ind = ind
    return worst_idx, worst_ind


def group_by_structure(population: list) -> dict[tuple[int, ...], list]:
    groups: dict[tuple[int, ...], list] = defaultdict(list)
    for ind in population:
        groups[structure_fingerprint(ind.attributeType)].append(ind)
    return groups


def restrict_same_structure_parents(parent_a, candidates: list, threshold: float) -> object | None:
    """
    Restricted mate selection: pick first candidate whose structural distance
    to parent_a is >= threshold. Returns None if none found.
    """
    for cand in candidates:
        if cand is parent_a:
            continue
        if structural_distance(parent_a.attributeType, cand.attributeType) >= threshold:
            return cand
    return None
