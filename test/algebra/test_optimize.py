from mechsynth.term import *
from mechsynth.symbolic.value import *
from mechsynth.symbolic.geom import *
from mechsynth.symbolic.object import *
from mechsynth.symbolic.relation import *
from mechsynth.symbolic.model import *
from mechsynth.symbolic.model_ops import *

from test.premade_models import test_data

from mechsynth.algebra.small_step import *
from mechsynth.algebra.util import *
from mechsynth.algebra.simplify import *
from mechsynth.algebra.sympy import *
from mechsynth.algebra.optimize import *

from pytest import * # type: ignore
import pytest
from pprint import *
from tabulate import tabulate

# TODO :: test_mval_hash_eq

def test_optimize_parse():
    """
    Just an empty test to make pytest force a parse check, of the various
    mechsynth.symbolic modules. 
    """
    assert True

@pytest.mark.parametrize("name,builder",**test_data())
def test_optimize_basic(name, builder):
    """
    See whether a model initializes and can build correctly.
    """

    model = Model(name)
    dirty = True
    printing = True
    counter = 1
    stats = list()

    with model.build():
        builder()

    if printing:
        print_graphs(f'opt_{name}_init', model)

    while dirty:

        print()

        dirty, new_model = model.run_algebra(
            OptimizeAlg(name=name,
                        counter=counter,
                        stats=stats,
                        num_steps=1))
        
        if printing: 
            print_graphs(f'opt_{name}_post({counter})', new_model)

        model = new_model
        counter += 1

    if printing:
        print_stats(f'opt_{name}', stats)
