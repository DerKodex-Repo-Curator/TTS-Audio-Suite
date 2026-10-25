import torch

from engines.moss_tts_cache_position import get_initial_cache_position


def test_initial_cache_position_starts_at_zero_without_past_cache() -> None:
    model_kwargs = get_initial_cache_position(4, torch.device("cpu"), {})

    assert torch.equal(model_kwargs["cache_position"], torch.tensor([0, 1, 2, 3]))


def test_initial_cache_position_skips_existing_legacy_cache_tokens() -> None:
    past_key_values = ((torch.empty(1, 1, 3, 1), torch.empty(1, 1, 3, 1)),)

    model_kwargs = get_initial_cache_position(
        5, torch.device("cpu"), {"past_key_values": past_key_values}
    )

    assert torch.equal(model_kwargs["cache_position"], torch.tensor([3, 4]))


def test_initial_cache_position_skips_existing_cache_object_tokens() -> None:
    class CacheWithSequenceLength:
        def get_seq_length(self) -> int:
            return 2

    model_kwargs = get_initial_cache_position(
        5, torch.device("cpu"), {"past_key_values": CacheWithSequenceLength()}
    )

    assert torch.equal(model_kwargs["cache_position"], torch.tensor([2, 3, 4]))


def test_initial_cache_position_preserves_a_caller_supplied_value() -> None:
    supplied = torch.tensor([7, 8])

    model_kwargs = get_initial_cache_position(
        4, torch.device("cpu"), {"cache_position": supplied}
    )

    assert model_kwargs["cache_position"] is supplied
