from mechsynth.term import *
from mechsynth.symbolic.value import *
from mechsynth.symbolic.geom import *
from mechsynth.symbolic.object import *
from mechsynth.symbolic.relation import *
from mechsynth.symbolic.model import *
from enum import Flag, auto
from mechsynth.algebra.small_step import *
from mechsynth.algebra.use_def import *
from mechsynth.algebra.util import *
from pytest import * # type: ignore
from graphviz import Digraph
from dataclasses import *
from enum import Enum, auto
import itertools
import subprocess
import sympy as sym

"""
This algebra translates elements into their sympy forms and uses that to
perform numerous simplification steps. 
"""

ASSERT = 'assert'
ASSUME = 'assume'
CONSTRAIN = 'constrain'
REQUIRE = 'req' 

@model_algebra
@dataclass
class SympyAlg():
    """
    We just run this algebra to gather sets of parameters, controls, and
    variables that are either *used by* or *defined over* each term.
    """

    dirty : bool = False
    relevant_terms : Dict[MVal,Set[str]] = field(default_factory=dict)
    tt_alg : TTypeAlg = field(default_factory=TTypeAlg)

    def _init_algebra(self, ctxt):
        ctxt.purge_keys({'sympy', 'sympy-eq'})
        self._add_relevant(ASSERT,    ctxt.assertions)
        self._add_relevant(ASSUME,    ctxt.assumptions)
        self._add_relevant(CONSTRAIN, ctxt.constraints)
        self._add_relevant(REQUIRE,   ctxt.guarantees)

    def _add_relevant(self, flag, terms):
        """
        Go through and add all the terms that we care about to the
        relevant terms list, add flags to keep track of which set they
        came from, and eventually belong in. 
        """
        for init_term in terms:

            init_term = id_to_mval(init_term)

            ts = set([init_term])

            # grab the equality set if needed
            if 'eq_set' in init_term and init_term['eq_set'] != None:
                ts.update(init_term['eq_set'])

            # Add flags and stuff 
            for t in ts:
                if t in self.relevant_terms:
                    self.relevant_terms[t].add(flag)
                else:
                    self.relevant_terms[t] = set([flag]) 

    def _init_pass(self, ctxt):
        pass

    def _end_pass(self, ctxt):
        return False

    def _end_algebra(self, ctxt):
        return None

    def _run(self, ident, val) -> None:
        # self.try_simp(ident, val)
        ident = id_to_mval(ident)
        if ident == ident.final: 
            self.try_eq_simp(ident, val)

    def try_simp(self, ident, val):

        ident = id_to_mval(ident)
        val = ident.val

        self.gen_rep(ident, val)

        # If there's no sympy term for this, there's nothing to do. 
        if (not 'sympy' in ident) or (ident['sympy'] == None):
            ident['sympy-simp'] = None
            return None

        expr = ident['sympy']
        simp = sym.simplify(expr)

        ident['sympy-simp'] = simp
        
    def try_eq_simp(self, ident, val):

        ident = id_to_mval(ident)
        val = ident.val

        self.gen_rep(ident, val)

        model = ident.parent_context

        if ((ident in self.relevant_terms)
            and ('sympy' in ident)
            and (ident['sympy'] != None)):
            # if we care about this term, we can attempt to solve for
            # each of the symbols within it.

            expr = ident['sympy']
            if isinstance(expr, sym.Rel):
                print(f'Found Relational')
                print(f'  Ident:{ident}')
                print(f'  expr :{expr}')
                print(f'  symbs:{expr.free_symbols}')
            elif (sym.Expr(expr).is_constant()):
                pass
            else:
                print(f'Not Relational')
                print(f'  flags:{self.relevant_terms[ident]}')
                print(f'  Ident:{ident}')
                print(f'  Ident:{ident.final}')
                print(f'  expr :{expr}')
                print(f'  symbs:{expr.free_symbols}')
                raise ValueError(f'Value not relational.')
        
        

    def gen_rep(self, ident, val) -> None:
        """
        Construct the sympy version of a formula. 
        """

        ident = id_to_mval(ident)
        val = ident.val

        self.tt_alg._run(ident, val)

        if ident['tensor_type'] == TType.SCALAR:

            for term in tlist(val):
                self.gen_rep(term, term.val)

            if type(val) in {Constant}:

                cv = val.const_val

                if type(cv) == bool:
                    if cv: 
                        ident['sympy'] = sym.BooleanTrue
                    else:
                        ident['sympy'] = sym.BooleanFalse
                elif type(cv) == int:
                    ident['sympy'] = sym.Integer(val.const_val)
                elif type(cv) == float:
                    ident['sympy'] = sym.Float(val.const_val)
                else:
                    raise TypeError

                if ident['sympy'] == None:
                    raise ValueError

            elif type(val) in {Param}:

                ident['sympy'] = sym.Symbol(val.name)

                if ident['sympy'] == None:
                    raise ValueError

            elif type(val) in {Control, Variable}:

                ident['sympy'] = sym.Symbol(val.name)

                if ident['sympy'] == None:
                    raise ValueError

            elif type(val) in {Mul, Add, Pow, Negate, Eq, Neq, And, Or, Not, Xor,
                               Implies, Mag, LessThan, LessThanEq,
                               GreaterThan, GreaterThanEq,
                               Sin, Cos, Tan, Asin, Acos, Atan, Atan2}:

                if all(('sympy' in t and (t['sympy'] != None)) for t in tlist(val)):

                    def get_sym(x):
                        s = x['sympy']
                        if s == 1 or s == 0:
                            s = sym.Integer(s)
                        return s

                    symval = tmap(get_sym, val)
                    fname = f'gen_{val.__class__.__name__.lower()}'

                    if hasattr(self, fname):
                        ident['sympy'] = getattr(self, fname)(ident, symval)
                    else:
                        raise NotImplementedError(f'need {fname}')
                       
            elif type(val) in {IfThenElse}:
                pass
            else:
                raise NotImplementedError(f'cannot sympyify {val}') 
        else:
            ident['sympy'] = None
            
    def gen_add(self, ident, val):
        return sym.Add(val.exp_a, val.exp_b)

    def gen_mul(self, ident, val):
        return sym.Mul(val.exp_a, val.exp_b)

    def gen_pow(self, ident, val):
        return sym.Pow(val.exp_a, val.exp_b)

    def gen_negate(self, ident, val):
        return sym.Mul(-1, val.exp)

    def gen_eq(self, ident, val):
        return sym.Eq(val.exp_a, val.exp_b)

    def gen_neq(self, ident, val):
        return sym.Ne(val.exp_a, val.exp_b)

    def gen_lessthan(self, ident, val):
        return sym.Lt(val.exp_a, val.exp_b)

    def gen_lessthaneq(self, ident, val):
        return sym.Le(val.exp_a, val.exp_b)

    def gen_greaterthan(self, ident, val):
        return sym.Gt(val.exp_a, val.exp_b)

    def gen_greaterthaneq(self, ident, val):
        return sym.Ge(val.exp_a, val.exp_b)

    def gen_and(self, ident, val):
        return sym.And(val.exp_a,val.exp_b)

    def gen_or(self, ident, val):
        return sym.Or(val.exp_a,val.exp_b)

    def gen_not(self, ident, val):
        return sym.Not(val.exp)

    def gen_xor(self, ident, val):
        return sym.Xor(val.exp_a,val.exp_b)

    def gen_implies(self, ident, val):
        return sym.Implies(val.exp_a,val.exp_b)

    def gen_mag(self, ident, val):
        return sym.abs(val.exp)

    def gen_sin(self, ident, val):
        return sym.sin(val.exp)

    def gen_cos(self, ident, val):
        return sym.cos(val.exp)

    def gen_tan(self, ident, val):
        return sym.tan(val.exp)

    def gen_asin(self, ident, val):
        return sym.asin(val.exp)

    def gen_acos(self, ident, val):
        return sym.acos(val.exp)

    def gen_atan(self, ident, val):
        return sym.atan(val.exp)

    def gen_atan2(self, ident, val):
        return sym.atan2(val.exp_a, val.exp_b)
