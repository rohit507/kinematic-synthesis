from mechsynth.term import *
from mechsynth.symbolic.value import *
from mechsynth.symbolic.geom import *
from mechsynth.symbolic.object import *
from mechsynth.symbolic.relation import *
from mechsynth.symbolic.model import *
from mechsynth.symbolic.model_ops import *

from mechsynth.algebra.small_step import TType,TTypeAlg
import dreal.symbolic as dreal

"""
This module is the algebra that converts a model into a dreal expression and
evaluates it.

This assumes that we're already at a form that has reduced out all the complex
relationships between 

In particular:
  - find parameters that satisfy the problem
  - given parameters check that variables are bound
  - given parameters and controls find all variables
"""

@model_algebra
class DRealAlg():
    """
    Figures out the tensor type of each term in our language, if we can tell.
    This lets us ensure that all of our final terms have the proper tensor
    type. 
    """
    def _init_algebra(self, ctxt):
        ctxt.purge_keys({'dreal_exp'})

    def _init_pass(self, ctxt):
        ctxt.run_algebra(TTypeAlg())
        pass

    def _end_pass(self, ctxt):
        return False

    def _end_algebra(self, ctxt):
        return None

    def _run(self, ident, val) -> None:
        """
        Basically, we can just do a simple translation of terms from our
        language to d-real's storing expressions as we go.

        Mind, dreal only supports expressions of scalars, so we use the
        ttype algebra to filter out higher level constraints.  
        """
        ident = id_to_mval(ident)

        if ident['tensor_type'] == TType.SCALAR:
            if hasattr(self, self.language[type(val)]):
                getattr(self, self.language[type(val)])(ident, val)
            else:
                raise Exception(f'Terminal of type {type(val)} is not convertible'
                              + " to dreal expression. Must only use scalar expressions.")

    def run_constant(self, ident, val) -> None:
        # if type(val.const_val) == bool:
        #     if val.const_cal:
        #     else:
        # else:
        ident['dreal_exp'] = val.const_val

    def run_param(self, ident, val) -> None:
        if val.v_type == ValType.BOOL:
            ident['dreal_exp'] = dreal.Variable(ident.uniq_name,
                                                dreal.Variable.Int) == 1.0
        elif val.v_type == ValType.REAL:
            ident['dreal_exp'] = dreal.Variable(ident.uniq_name,
                                                dreal.Variable.Real)
        elif val.v_type == ValType.INT:
            ident['dreal_exp'] = dreal.Variable(ident.uniq_name,
                                                dreal.Variable.Int)

    def run_control(self, ident, val) -> None:
        ident['dreal_exp'] = dreal.Variable(ident.uniq_name,dreal.Variable.Real)

    def run_ifthenelse(self, ident, val) -> None:
        vc = val.exp_cond['dreal_exp']
        vt = val.exp_true['dreal_exp']
        vf = val.exp_false['dreal_exp']

        ident['dreal_exp'] = dreal.if_then_else(vc, vt, vf)
    
    def run_eq(self, ident, val) -> None:
        a = val.exp_a['dreal_exp']
        b = val.exp_b['dreal_exp']

        ident['dreal_exp'] = a == b

    def run_greaterthan(self, ident, val) -> None:
        a = val.exp_a['dreal_exp']
        b = val.exp_b['dreal_exp']

        ident['dreal_exp'] = a > b

    def run_greaterthaneq(self, ident, val) -> None:
        a = val.exp_a['dreal_exp']
        b = val.exp_b['dreal_exp']

        ident['dreal_exp'] = a >= b

    def run_lessthan(self, ident, val) -> None:
        a = val.exp_a['dreal_exp']
        b = val.exp_b['dreal_exp']

        ident['dreal_exp'] = a < b

    def run_lessthaneq(self, ident, val) -> None:
        a = val.exp_a['dreal_exp']
        b = val.exp_b['dreal_exp']

        ident['dreal_exp'] = a <= b

    def run_add(self, ident, val) -> None:
        a = val.exp_a['dreal_exp']
        b = val.exp_b['dreal_exp']

        ident['dreal_exp'] = a + b

    def run_mul(self, ident, val) -> None:
        a = val.exp_a['dreal_exp']
        b = val.exp_b['dreal_exp']

        ident['dreal_exp'] = a * b

    def run_and(self, ident, val) -> None:
        a = val.exp_a['dreal_exp']
        b = val.exp_b['dreal_exp']

        ident['dreal_exp'] = dreal.logical_and(a,b)

    def run_or(self, ident, val) -> None:
        a = val.exp_a['dreal_exp']
        b = val.exp_b['dreal_exp']

        ident['dreal_exp'] = dreal.logical_or(a,b)

    def run_sin(self, ident, val) -> None:
        ident['dreal_exp'] = dreal.sin(val.exp['dreal_exp'])

    def run_cos(self, ident, val) -> None:
        ident['dreal_exp'] = dreal.cos(val.exp['dreal_exp'])

    def run_tan(self, ident, val) -> None:
        ident['dreal_exp'] = dreal.tan(val.exp['dreal_exp'])


