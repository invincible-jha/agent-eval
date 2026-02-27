"""Tests for agent_eval.adversarial.robustness_tiers."""
from __future__ import annotations

import pytest

from agent_eval.adversarial.robustness_tiers import (
    AdversarialInput,
    RobustnessTier,
    TIER_DEFINITIONS,
    TierDefinition,
    generate_all_tier_inputs,
    generate_tier_inputs,
)


class TestRobustnessTier:
    def test_all_tiers_exist(self) -> None:
        tiers = list(RobustnessTier)
        assert len(tiers) == 5

    def test_tier_ordering(self) -> None:
        assert RobustnessTier.T1_NAIVE < RobustnessTier.T5_ADAPTIVE

    def test_tier_values_ascending(self) -> None:
        values = [tier.value for tier in RobustnessTier]
        assert values == sorted(values)


class TestTierDefinitions:
    def test_all_tiers_defined(self) -> None:
        for tier in RobustnessTier:
            assert tier in TIER_DEFINITIONS

    def test_definition_has_pattern_classes(self) -> None:
        for tier_def in TIER_DEFINITIONS.values():
            assert len(tier_def.pattern_classes) > 0

    def test_definition_has_description(self) -> None:
        for tier_def in TIER_DEFINITIONS.values():
            assert len(tier_def.description) > 10

    def test_tier_names_unique(self) -> None:
        names = [td.name for td in TIER_DEFINITIONS.values()]
        assert len(names) == len(set(names))


class TestGenerateTierInputs:
    def test_generates_inputs_for_t1(self) -> None:
        inputs = generate_tier_inputs(RobustnessTier.T1_NAIVE)
        assert len(inputs) > 0

    def test_all_inputs_correct_tier(self) -> None:
        for tier in RobustnessTier:
            inputs = generate_tier_inputs(tier)
            for inp in inputs:
                assert inp.tier == tier

    def test_inputs_have_content(self) -> None:
        for tier in RobustnessTier:
            inputs = generate_tier_inputs(tier)
            for inp in inputs:
                assert inp.content.strip() != ""

    def test_input_ids_unique_within_tier(self) -> None:
        for tier in RobustnessTier:
            inputs = generate_tier_inputs(tier)
            ids = [inp.input_id for inp in inputs]
            assert len(ids) == len(set(ids))

    def test_pattern_classes_covered(self) -> None:
        for tier in RobustnessTier:
            inputs = generate_tier_inputs(tier)
            generated_classes = {inp.pattern_class for inp in inputs}
            defined_classes = set(TIER_DEFINITIONS[tier].pattern_classes)
            assert generated_classes == defined_classes

    def test_seed_reproducibility(self) -> None:
        inputs_a = generate_tier_inputs(RobustnessTier.T1_NAIVE, seed=42)
        inputs_b = generate_tier_inputs(RobustnessTier.T1_NAIVE, seed=42)
        contents_a = [inp.content for inp in inputs_a]
        contents_b = [inp.content for inp in inputs_b]
        assert contents_a == contents_b

    def test_different_seeds_may_differ(self) -> None:
        inputs_a = generate_tier_inputs(RobustnessTier.T1_NAIVE, seed=1)
        inputs_b = generate_tier_inputs(RobustnessTier.T1_NAIVE, seed=99)
        contents_a = [inp.content for inp in inputs_a]
        contents_b = [inp.content for inp in inputs_b]
        # May differ (not guaranteed due to limited topics/templates)
        # Just verify both are non-empty and valid
        assert len(contents_a) > 0
        assert len(contents_b) > 0

    def test_custom_topics_injected(self) -> None:
        custom_topic = "UNIQUE_TOPIC_MARKER_XYZ"
        inputs = generate_tier_inputs(
            RobustnessTier.T1_NAIVE,
            topics=[custom_topic],
        )
        assert all(custom_topic in inp.content for inp in inputs)

    def test_expected_defense_field_set(self) -> None:
        inputs = generate_tier_inputs(RobustnessTier.T1_NAIVE)
        for inp in inputs:
            assert inp.expected_defense != ""


class TestGenerateAllTierInputs:
    def test_returns_all_tiers(self) -> None:
        all_inputs = generate_all_tier_inputs()
        for tier in RobustnessTier:
            assert tier in all_inputs
            assert len(all_inputs[tier]) > 0

    def test_total_input_count(self) -> None:
        all_inputs = generate_all_tier_inputs()
        total = sum(len(inputs) for inputs in all_inputs.values())
        # 5 tiers * 3 pattern classes * 3 templates = 45
        assert total == 45


class TestAdversarialInputDataclass:
    def test_frozen(self) -> None:
        inp = AdversarialInput(
            tier=RobustnessTier.T1_NAIVE,
            input_id="t1_test_000",
            content="test content",
            pattern_class="direct_override",
        )
        with pytest.raises((AttributeError, TypeError)):
            inp.content = "changed"  # type: ignore[misc]
