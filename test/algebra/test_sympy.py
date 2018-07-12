from mechsynth.term import *
from mechsynth.symbolic.value import *
from mechsynth.symbolic.geom import *
from mechsynth.symbolic.object import *
from mechsynth.symbolic.relation import *
from mechsynth.symbolic.model import *
from mechsynth.symbolic.model_ops import *

import test.premade_models
from mechsynth.algebra.use_def import *
from pytest import * # type: ignore
import pytest
import pprint

def test_sympy_parse():
    """
    Just an empty test to make pytest force a parse check, of the various
    mechsynth.symbolic modules. 
    """
    assert True

# NOTE :: all the actual testing is done in test_optimize which is better 
