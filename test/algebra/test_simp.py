from mechsynth.term import *
from mechsynth.symbolic.value import *
from mechsynth.symbolic.geom import *
from mechsynth.symbolic.object import *
from mechsynth.symbolic.relation import *
from mechsynth.symbolic.model import *
from mechsynth.symbolic.model_ops import *

from test.premade_models import *
from mechsynth.algebra.small_step import *
from mechsynth.algebra.util import *
from mechsynth.algebra.simplify import *
from pytest import * # type: ignore
import pytest
import pprint
import pathlib


# TODO :: test_mval_hash_eq

def test_simp_parse():
    """
    Just an empty test to make pytest force a parse check.
    """
    assert True

def dprint(*args, **kwargs):
    debug = True
    if debug:
        print(*args, **kwargs)


def model_stats(model): 
    dprint()
    dprint(f'  Stats:')
    dprint(f'     terms   : {len(model._id_map)}')
    dprint(f'     frames  : {len(model.ref_frames)}')
    dprint(f'     objs    : {len(model.objects)}')
    dprint(f'     params  : {len(model.parameters)}')
    dprint(f'     controls: {len(model.controls)}')
    dprint(f'     vars    : {len(model.variables)}')
    dprint(f'     asserts : {len(model.assertions)}')
    dprint(f'     assums  : {len(model.assumptions)}')
    dprint(f'     consts  : {len(model.constraints)}')
    dprint(f'     reqs    : {len(model.guarantees)}')
    dprint(f'     costs   : {len(model.cost_terms)}')
    dprint()

@pytest.mark.parametrize("name,builder",**test_data())
def test_simp_basic(name, builder):
    """
    See whether a model initializes and can build correctly.
    """

    model = Model("Test Model")
    dirty = True
    printing = True
    counter = 1

    with model.build():
        builder()

    if printing: 
        print_graphs(f'_simp_{name}_initial',model)
    dprint()

    model_stats(model)
        
    while dirty:

        dprint()
        dprint()

        dprint(f'Running Small Step Algebra ({counter})') 
        ssalg = SmallStepAlg(max_runs = 3)
        model.run_algebra(ssalg)
        dirty = ssalg.dirty
        model_stats(model)

        dprint(f'Running Equality Elimination Algebra ({counter})')
        eealg = EqElimAlg()
        model.run_algebra(eealg)
        dirty = dirty or eealg.dirty
        model_stats(model)

        if printing: 
            print_graphs(f'_simp_{name}_{counter}_pre_const',model)
        dprint(f'Running Const Prop Algebra ({counter})')
        cpalg = ConstPropAlg()
        model.run_algebra(cpalg)
        dirty = dirty or cpalg.g_dirty

        #dprint(f'Running Finishing Algebra ({counter})')
        #model.run_algebra(FinalizeAlg())

        model_stats(model)
        if printing: 
            print_graphs(f'_simp_{name}_{counter}',model)

        dprint(f'Pruning ({counter})')
        palg = PruneAlg(f'{name}_loop_{counter}', keep_relevant=False)
        model = model.run_algebra(palg)

        model_stats(model)

        if printing:
            print_graphs(f'_simp_{name}_{counter}_pruned',model)

        dirty = dirty or palg.dirty
        counter += 1
