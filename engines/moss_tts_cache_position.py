import torch


def get_initial_cache_position(cur_len, device, model_kwargs):
    """Set the initial token positions used by MOSS generation."""
    if model_kwargs.get("cache_position") is not None:
        return model_kwargs

    cache_position = torch.arange(cur_len, dtype=torch.long, device=device)
    past_key_values = model_kwargs.get("past_key_values")
    if past_key_values is not None:
        get_seq_length = getattr(past_key_values, "get_seq_length", None)
        if callable(get_seq_length):
            past_length = get_seq_length() or 0
        elif past_key_values:
            past_length = past_key_values[0][0].shape[2]
        else:
            past_length = 0
        cache_position = cache_position[past_length:]

    model_kwargs["cache_position"] = cache_position
    return model_kwargs
