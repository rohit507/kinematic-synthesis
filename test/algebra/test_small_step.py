from mechsynth.term import *
from mechsynth.symbolic.value import *
from mechsynth.symbolic.geom import *
from mechsynth.symbolic.object import *
from mechsynth.symbolic.relation import *
from mechsynth.symbolic.model import *
from mechsynth.symbolic.model_ops import *

from test.premade_models import *
from mechsynth.algebra.small_step import *
from pytest import * # type: ignore
import pytest
import pprint
import pathlib


# TODO :: test_mval_hash_eq

def test_small_step_parse():
    """
    Just an empty test to make pytest force a parse check.
    """
    assert True



@pytest.mark.parametrize("name,builder",**test_data())
def test_small_step_basic(name, builder):
    """
    See whether a model initializes and can build correctly.
    """

    model = Model("Test Model")
    alg   = SmallStepAlg()

    with model.build():
        builder()

    model.run_algebra(alg)

