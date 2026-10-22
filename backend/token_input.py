"""Normalize one tokenizer result without decoding or silently batching IDs."""
from collections.abc import Mapping


def token_ids(value):
    if isinstance(value, Mapping):
        value = value.get('input_ids')
    if not isinstance(value, list) or not value or any(type(i) is not int or i < 0 for i in value):
        raise ValueError('Expected a nonempty flat list of nonnegative token IDs')
    return list(value)
