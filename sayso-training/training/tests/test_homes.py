
from __future__ import annotations

import collections
import random

from generators.homes import generate_home


def test_home_sizes_respected() -> None:
    rng = random.Random(42)
    for size in (8, 16, 32):
        home = generate_home(1, size, rng)
        assert len(home["entities"]) >= size


def test_home_has_sayso_area() -> None:
    home = generate_home(5, 16, random.Random(7))
    assert home["sayso_entity_area"]
    assert all(e["area"] for e in home["entities"])


def test_entity_ids_match_domains() -> None:
    home = generate_home(3, 16, random.Random(11))
    for entity in home["entities"]:
        assert entity["entity_id"].startswith(entity["domain"] + ".")


def test_entity_ids_are_unique_within_a_home() -> None:
    for size in (16, 32, 64):
        home = generate_home(9, size, random.Random(size))
        ids = [entity["entity_id"] for entity in home["entities"]]
        assert len(ids) == len(set(ids))


def test_entity_names_balance_realistic_reuse_and_variety() -> None:
    names: list[str] = []
    rng = random.Random(20260906)
    for index, size in enumerate([16, 32, 64] * 100):
        names.extend(entity["name"].casefold() for entity in generate_home(index, size, rng)["entities"])
    counts = collections.Counter(names)
    assert len(counts) >= 1000, "fixture/room variety collapsed"
    assert max(counts.values()) / len(names) < 0.03, "one name dominates the corpus"
    familiar = [name for name, count in counts.items() if count >= 3
                and name.startswith(("kitchen ", "living room ", "bathroom ", "master bedroom "))]
    assert len(familiar) >= 20, "ordinary room/device names should recur across homes"


def test_generator_cannot_emit_an_eval_entity_name() -> None:
    from evals.cases import entity_names_for_tag

    eval_names = entity_names_for_tag("gold", include_aliases=False) | entity_names_for_tag(
        "shadow", include_aliases=False
    )
    generated: set[str] = set()
    rng = random.Random(11)
    for index, size in enumerate([32, 64] * 120):
        generated.update(entity["name"] for entity in generate_home(index, size, rng)["entities"])
    assert not generated & eval_names, sorted(generated & eval_names)


def _all_target_names(home) -> set[str]:
    names: set[str] = set()
    for entity in home["entities"]:
        names.add(entity["name"].casefold())
        names.update(alias.casefold() for alias in entity.get("aliases", []))
    return names


def test_context_entities_present_and_noncolliding() -> None:
    from generators.homes import _eval_entity_names

    rng = random.Random(99)
    for index, size in enumerate([16, 32, 64] * 20):
        home = generate_home(index, size, rng)
        context = home["context_entities"]
        assert context, f"home {index}/{size} has no context entities"
        for item in context:
            assert set(item) <= {"name", "domain", "area"}
            assert item["domain"] in {
                "sensor", "binary_sensor", "person", "sun", "select",
            }, f"{item['name']} has targetable domain {item['domain']}"
        target_names = _all_target_names(home)
        reserved = _eval_entity_names()
        context_names = [item["name"].casefold() for item in context]
        assert len(context_names) == len(set(context_names)), f"duplicate context names in home {index}"
        assert not (set(context_names) & target_names), f"context name collision in home {index}"
        assert not (set(context_names) & reserved), f"eval name leaked into context in home {index}"


def test_context_entities_scale_with_home_size() -> None:
    for index, size in [(1, 16), (2, 32), (3, 64)]:
        home = generate_home(index, size, random.Random(5))
        ratio = len(home["context_entities"]) / len(home["entities"])
        assert 0.1 <= ratio < 1.25, f"size {size}: {ratio:.2f}"


def test_context_entities_are_seed_deterministic() -> None:
    first = generate_home(7, 32, random.Random(1))["context_entities"]
    second = generate_home(7, 32, random.Random(1))["context_entities"]
    assert first == second


def test_apostrophe_names_stay_represented_but_not_dominant() -> None:
    names: list[str] = []
    rng = random.Random(4)
    for index, size in enumerate([16, 32] * 60):
        names.extend(entity["name"] for entity in generate_home(index, size, rng)["entities"])
    share = sum("'" in name for name in names) / len(names)
    assert 0.05 < share < 0.25, share
