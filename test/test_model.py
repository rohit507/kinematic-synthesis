from mechsynth.term import *
from mechsynth.symbolic.value import *
from mechsynth.symbolic.geom import *
from mechsynth.symbolic.object import *
from mechsynth.symbolic.relation import *
from mechsynth.symbolic.model import *
from mechsynth.symbolic.model_ops import *
from mechsynth.algebra.use_def import *
from pytest import * # type: ignore

# TODO :: test_mval_hash_eq

def test_model_parse():
    """
    Just an empty test to make pytest force a parse check, of the various
    mechsynth.symbolic modules. 
    """
    assert True

def test_model_basic():
    """
    See whether a model initializes and can build correctly.
    """

    model = Model("Test Model")

    with model.build():
        ang = control("theta")

        assume(ang > -3)
        assume(ang < 3)

        pt = mat3x1(sin(ang), cos(ang), 0.0)

        a = parameter('a',bool)
        b = parameter('b',bool)
        c = parameter('c',bool)
        d = parameter('d',bool)

        print(a)
        x = if_then_else(a,sin(ang),cos(ang)) * if_then_else(b, 1.0, -1.0)
        y = if_then_else(c,sin(ang),cos(ang)) * if_then_else(d, 1.0, -1.0)

        op = mat3x1(x,y, 0.0)

        require(dot(pt,op).equals(0.0))

    print(model.to_graphviz().source)
