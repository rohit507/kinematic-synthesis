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

# TODO :: test_mval_hash_eq

def test_use_def_parse():
    """
    Just an empty test to make pytest force a parse check, of the various
    mechsynth.symbolic modules. 
    """
    assert True

def fst(a): return a[0]

def print_model_use_def(model): 
    for i in model.topolist():
        print(i.assign_statement)
        print(f'   use-typ: {i["use_type"]}')
        print(f'   uses   :')
        for u in i['uses']:
            print(f'      {u.uniq_name}')
        print(f'   defines:')
        for u in i['defines']:
            print(f'      {u.uniq_name}')

@pytest.mark.parametrize("name,builder",test.premade_models.test_data,ids=fst)
def test_use_def_basic(name, builder):
    """
    See whether a model initializes and can build correctly.
    """

    model = Model("Test Model")
    alg   = UseDefAlg()

    with model.build():
        builder()

    model.run_algebra(alg)

