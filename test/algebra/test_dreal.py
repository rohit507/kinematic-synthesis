from mechsynth.term import *
from mechsynth.symbolic.value import *
from mechsynth.symbolic.geom import *
from mechsynth.symbolic.object import *
from mechsynth.symbolic.relation import *
from mechsynth.symbolic.model import *
from mechsynth.symbolic.model_ops import *

import test.premade_models
from mechsynth.algebra.small_step import *
from mechsynth.algebra.dreal import *
from pytest import * # type: ignore
import pytest
import pprint
import dreal
from dreal.symbolic import *
from dreal.api import CheckSatisfiability, Minimize

def test_dreal_parse():
    """
    Just an empty test to make pytest force a parse check.
    """
    
# TODO :: Find a value of c such that x^2 + c > d for x in range 
# def test_dreal_interface_simple(): 
    
def test_dreal_interface_othro():
    """
    Just an empty test to make pytest force a parse check.
    """
    t = Variable('theta',Variable.Real)
    a = Variable('a',Variable.Bool)
    b = Variable('b',Variable.Bool)
    c = Variable('c',Variable.Bool)
    d = Variable('d',Variable.Bool)


    fun = forall([t], logical_imply(logical_and(t < 3,t > -3),
                                      (((sin(t)
                                         * if_then_else(b == True, 1 , -1)
                                         * if_then_else(a == True,sin(t),cos(t)))
                                        + (cos(t)
                                           * if_then_else(d == True, 1 , -1)
                                           * if_then_else(c == True,sin(t),cos(t))))
            == 0)))

    result = CheckSatisfiability(fun, 0.01)
    print(result)
    assert False

def test_boolean_if_then():
    """
    Tests whether boolean constraints work. True is '1', false is '0'.
    """

    a = Variable('a',Variable.Int)
    b = Variable('b',Variable.Int)

    fun = logical_and(if_then_else(a == False, 20, -20) > 0,
                      if_then_else(b == True, 13, -12) > 0)

    result = CheckSatisfiability(fun, 0.01)
    print(result)


def test_nested_forall():
    """
    Test whether we can choose parameters for a system, such that there is a
    satisfying solution over an entire range of values. 

    Basically, choose an origin (ox,oy) and the lengths of two arms (l1 & l2)
    so that there is an angle (t1 & t2) for each arm, that allows it to reach
    any point within a circle centered at (25,25) with radius 4.

    We want to make sure that the lengths and origin the tool chooses allows
    for 
    
    >   there should:
    >     exists. l1 in (0,20) , l2 (0,20), ox (-20,20), oy (-20,20) 
    >
    >   such that:
    >     for all. px in (20, 30), py in (20,30)
    >
    >   given assumptions:
    >     sqrt((25 - px)**2 + (25 - py)**2) <= 4
    >
    >   there should:
    >     exists. t1 in (-pi, pi), t2 in (-pi,pi)
    >
    >   with constraints:
    >      t2 > t1
    >
    >   that meets the requirements:
    >         (px == l2*sin(t1) + l2*sin(t2) + ox) 
    >      && (py == l1*cos(t1) + l2*cos(t2) + oy)

    """

    # Parameters 
    l1 = Variable('l1',Variable.Real)
    l2 = Variable('l2',Variable.Real)
    ox = Variable('ox',Variable.Real)
    oy = Variable('oy',Variable.Real)

    param_bounds = logical_and(l1 > 0, l1 < 20,
                               l2 > 0, l2 < 20,
                               ox > -20, ox < 20,
                               oy > -20, oy < 20)

    # Independent Variables
    px = Variable('px',Variable.Real)
    py = Variable('py',Variable.Real)

    ivar_bounds = logical_and(px > 20, px < 30,
                              py > 20, py < 30)

    ivar_assum = sqrt((25 - px)**2 + (25 - py)**2) < 4

    # Dependent Variables
    t1 = Variable('t1',Variable.Real)
    t2 = Variable('t2',Variable.Real)

    dvar_bounds = logical_and(t1 >= -3.14, t1 <= 3.14,
                              t2 >= -3.14, t2 <= 3.14,
                              t1 >= t2)

    req = logical_and(px == l1*sin(t1) + l2*sin(t2) + ox,
                      py == l1*cos(t1) + l2*cos(t2) + oy)



    def exists(vs, fun):
        return logical_not(forall(vs, logical_not(fun)))



    fun = logical_and(param_bounds,
                      forall([px,py],
                             logical_imply(logical_and(ivar_bounds, ivar_assum),
                                           exists([t1,t2],
                                                  logical_and(dvar_bounds,req)))))

    result = CheckSatisfiability(fun, 0.01)
    print(result)
    assert False


def fst(a): return a[0]

def print_model_trans(model): 
    for t in model.topolist():
        m = id_to_mval(t)
        print(m.assign_statement)
        if m.has_next:
            if m['next'] != None:
                print("    Next  : " + m['next'].assign_statement)
            print("    Final : " + m.final.assign_statement)

def model_graraphviz_trans(model):
    g = model.to_graphviz()
    for t in model.topolist():
        m = id_to_mval(t)
        g.edge(m.short_name, m.final.short_name, color="blue")
    return g

@pytest.mark.parametrize("name,builder",test.premade_models.test_data,ids=fst)
def test_dreal_basic(name, builder):
    """
    See whether a model initializes and can build correctly.
    """

    model = Model("Test Model")
    ssalg = SmallStepAlg()
    dralg = DRealAlg() 

    with model.build():
        builder()

    model.run_algebra(ssalg)
    model.run_algebra(dralg)

def test_dreal_orthogonal_sat():
    """
    See whether a model initializes and can build correctly.
    """

    model = Model("Test Model")
    ssalg = SmallStepAlg()
    dralg = DRealAlg() 

    with model.build():
        test.premade_models.find_orthogonal()

    model.gen_output_terms()
    model.run_algebra(ssalg)
    model.run_algebra(dralg)

    print(model_graraphviz_trans(model).source())

    controls = list()

    for v in model.controls:
        controls.append(v.final['dreal_exp'])

    fun = forall(controls, logical_imply(logical_and(
        model.assumption_term().final['dreal_exp'],
        model.constraint_term().final['dreal_exp']), 
        model.guarantee_term().final['dreal_exp']))

    print(fun)
    result = CheckSatisfiability(fun, 0.01)
    print(result)

