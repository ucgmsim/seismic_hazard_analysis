import numpy as np
from hypothesis import given, strategies as st, settings, Verbosity


@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1))
def test_np_sort_numerical_arrays(input_list: list):
    arr = np.array(input_list)
    sorted_arr = np.sort(arr)

    # Check that each element is >= the previous element
    assert np.all(np.diff(sorted_arr) >= 0)

    # Check that sorted array has same length as original
    assert len(sorted_arr) == len(arr)

    # Check that sorted array contains all original elements
    assert np.all(np.unique(arr) == np.unique(sorted_arr))
    assert np.all(
        np.unique(arr, return_counts=True)[1]
        == np.unique(sorted_arr, return_counts=True)[1]
    )


def pga_model(mag: float, rrup: float, vs30: float):
    return 1


@given(
    st.floats(min_value=0, max_value=9.0),
    st.floats(min_value=0, max_value=500),
    st.floats(min_value=0, max_value=1500),
)
@settings(verbosity=Verbosity.verbose)
def test_pga_model(mag: float, rrup: float, vs30: float):
    model_output = pga_model(mag, rrup, vs30)
    assert model_output >= 0
