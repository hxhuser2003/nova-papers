from collections import UserDict
import pytest
from backend.token_input import token_ids

def test_batchencoding_like_mapping_extracts_ids():
    assert token_ids(UserDict(input_ids=[1, 2], attention_mask=[1, 1])) == [1, 2]

def test_list_preserves_ids():
    assert token_ids([3, 4]) == [3, 4]

@pytest.mark.parametrize('value', [[[1, 2]], [True], [-1], {'attention_mask': [1]}, 'text'])
def test_bad_shapes_fail(value):
    with pytest.raises(ValueError):
        token_ids(value)
