import random

import pytest

from exp.data import render
from exp.prompts import ALIAS_CATALOG_MARKER, load_sop_prompt, render_sop_prompt
from exp.world.aliases import REGION_ALIASES, SERVICE_ALIASES
from exp.world.services import REGIONS, SERVICES


def test_generator_and_sop_share_alias_objects():
    assert render.SERVICE_ALIASES is SERVICE_ALIASES
    assert render.REGION_ALIASES is REGION_ALIASES
    assert set(SERVICE_ALIASES) == set(SERVICES)
    assert set(REGION_ALIASES) == set(REGIONS)


@pytest.mark.parametrize("catalog", [SERVICE_ALIASES, REGION_ALIASES])
def test_aliases_are_unique_across_canonical_values(catalog):
    owners = {}
    for canonical, names in catalog.items():
        assert canonical in names
        for name in names:
            normalized = name.casefold()
            assert normalized not in owners or owners[normalized] == canonical
            owners[normalized] = canonical


@pytest.mark.parametrize("canonical,aliases", list(SERVICE_ALIASES.items()) + list(REGION_ALIASES.items()))
def test_sop_contains_every_generator_alias(canonical, aliases):
    prompt = load_sop_prompt()
    expected_row = f"| `{canonical}` | {', '.join(f'`{alias}`' for alias in aliases)} |"
    assert expected_row in prompt
    assert ALIAS_CATALOG_MARKER not in prompt


def test_catalog_updates_are_rendered_without_editing_the_template(monkeypatch):
    before = load_sop_prompt()
    monkeypatch.setitem(SERVICE_ALIASES, "cart-svc", SERVICE_ALIASES["cart-svc"] + ["test-basket-alias"])
    after = load_sop_prompt()
    assert "test-basket-alias" not in before
    assert "`test-basket-alias`" in after
    assert before != after


@pytest.mark.parametrize("template", ["no marker", ALIAS_CATALOG_MARKER * 2])
def test_sop_rejects_missing_or_duplicate_catalog_marker(template):
    with pytest.raises(ValueError, match="exactly one"):
        render_sop_prompt(template)


def test_sop_preserves_missing_and_ambiguous_field_policy():
    prompt = load_sop_prompt()
    assert "adjacent-letter transposition" in prompt
    assert "unique catalog service" in prompt
    assert "outside the catalog" in prompt
    assert "multiple services" in prompt
    assert "missing, unsupported, or ambiguous" in prompt
    assert "never infer it from the service or a timestamp" in prompt
    assert "DO NOT convert between relative time and absolute timestamp" in prompt


def test_alias_refactor_does_not_change_seeded_alert_generation():
    alert = render.render_alert_templated(
        service="cart-svc", symptom="high_cpu", region="us-west", started_at=None,
        started_minutes_ago=12, received_at="2026-10-22T12:00:00Z",
        style="slack_message", typo_level=0, rng=random.Random(0),
    )
    assert alert == (
        "Received: 2026-10-22T12:00:00Z\n"
        "@channel heads up: basket-svc is seeing cpu load high in us-west "
        "(started about 12 min ago). Investigating."
    )
