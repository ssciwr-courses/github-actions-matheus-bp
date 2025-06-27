import numpy as np
import transform as tf
import pytest


@pytest.mark.parametrize("myinput, myref", 
[(1, np.pi), (0, 0), (2.1, np.pi * 2.1**2)])
def test_area_circ(myinput, myref):
    assert tf.area_circ(myinput) == myref


def test_area_circ_valueerror():
    with pytest.raises(ValueError):
        tf.area_circ(-1)
