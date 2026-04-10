import numpy as np
import pytest

from pysurgery.core.exact_algebra import coerce_int_matrix, normalize_word_token, validate_group_descriptor


def test_coerce_int_matrix_accepts_integer_valued_float_entries():
    m = coerce_int_matrix([[1.0, 2.0], [3.0, 4.0]])
    assert m.dtype == np.int64
    assert np.array_equal(m, np.array([[1, 2], [3, 4]], dtype=np.int64))


def test_coerce_int_matrix_rejects_non_integer_entries():
    with pytest.raises(ValueError):
        coerce_int_matrix([[1.5, 0.0]])


def test_normalize_word_token_handles_inverse_suffix_forms():
    assert normalize_word_token("g_1-1") == "g_1^-1"
    assert normalize_word_token("g_1^-1") == "g_1^-1"


def test_validate_group_descriptor_recognizes_supported_forms():
    assert validate_group_descriptor("1")[0]
    assert validate_group_descriptor("Z")[0]
    assert validate_group_descriptor("Z_7")[0]
    assert validate_group_descriptor("Z x Z_3")[0]
    assert not validate_group_descriptor("PSL_2(7)")[0]

