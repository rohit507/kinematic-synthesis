
from mechsynth.term import *
from mechsynth.symbolic.value import *
from mechsynth.symbolic.geom import *
from mechsynth.symbolic.object import *
from mechsynth.symbolic.relation import *
from mechsynth.symbolic.model import *
from mechsynth.symbolic.model_ops import *
from mechsynth.algebra.small_step import *
from mechsynth.algebra.use_def import *
from mechsynth.algebra.util import *
from pytest import * # type: ignore
from graphviz import Digraph
from dataclasses import *
from enum import Enum, auto
import itertools
import subprocess


"""
This is the algebra we use to simplify terms within the language,
a combination of constant propagation and algebraic simplification from
sympy. 
"""

def dprint(*args, **kwargs):
    debug = False
    if debug:
        print(*args, **kwargs)


@model_algebra
class EqElimAlg():
    """
    This algebra goes through the various assertions, assumption, and
    constraints, so that it gathers up any sets of equal elements in the
    context, and updates the next term. 
    """

    dirty : bool = False

    def _init_algebra(self, ctxt):

        facts = set()

        # Gather all the facts. 
        for term in ctxt.assumptions:
            facts.update(AssocAlg.get(ALL, term.final))

        for term in ctxt.constraints:
            facts.update(AssocAlg.get(ALL, term.final))

        for term in ctxt.guarantees:
            facts.update(AssocAlg.get(ALL, term.final))

        dprint(f'Facts Size              : {len(facts)}')

        # Prune out any which are not scalar

        new_facts = set()

        ctxt.run_algebra(TTypeAlg())

        for f in facts:
            if f['tensor_type'] == TType.SCALAR:
                new_facts.add(f)

        facts = new_facts
        dprint(f'Facts Size (scalar only): {len(facts)}')

        # Prune out any which are not equalities
        new_facts = set()

        for f in facts:
            if type(f.val) == Eq:
                new_facts.add(f)

        facts = new_facts 
        dprint(f'Facts Size (eq only)    : {len(facts)}')

        # add equality terms using set_eq
        for f in facts:
            val = f.val

            # dprint(f' setting two vals equivalent')
            # dprint(val.exp_a)
            # dprint(val.exp_b)


            val.exp_a.set_eq(val.exp_b)
            val.exp_b.set_eq(val.exp_a)

        ctxt.run_algebra(UseDefAlg())

    def _init_pass(self, ctxt):
        self.dirty = False 

    def _end_pass(self, ctxt):
        return False

    def _end_algebra(self, ctxt):
        pass
        # if self.dirty:
        #     print("propagating all updates")
        #     falg = FinalizeAlg()
        #     ctxt.run_algebra(falg)
        #     self.dirty = self.dirty or falg.dirty

    def _run(self, ident, val):
        ident = id_to_mval(ident)

        if ident.has_next:
            return None

        ident = ident.final
        val = tmap(lambda x: id_to_mval(x).final, ident.val)

        if ('eq_set') in ident and (type(val) in {Variable, Param}):
                # if we have a variable we want to choose a minimal term
                # but preserve all the equality relations that are around.
                replacement = None
                eq_set = set()
                uses = None
                for term in ident['eq_set']:
                    term = term.final
                    # we don't really car about option choice here as long as
                    # we strictly prefer terms that don't actually depend on
                    # This variable. 
                    if ((uses == None) or (term['uses'] < uses)) or (
                            (ident in uses) and (not (ident in term['uses']))):
                        uses = term['eq_set']
                        replacement = term

                    # And we want to add things to the new EQ set that are
                    # strictly better than their competitors.
                    # NOTE :: Theoretically this pruning could make it harder
                    #        for sympy to actually do simplifications, so
                    #        we're dropping it for now. 
                    # for ot in ident['eq_set']:
                    #     ot = ot.final
                    #     if term['uses'] < ot['uses']:
                    #         eq_set.remove(ot)
                    eq_set.add(term) 

                # Add in our replacement and equalities, making sure to gather
                # all of the remaining
                if replacement != None:
                    ident.set_next(replacement, color="deeppink1")
                    for term in eq_set:
                        if term != replacement:
                            replacement.set_eq(term) 
                    self.dirty = True
                    dprint(f'replacing variable {ident.uniq_name} with {replacement.val}')

@model_algebra
class ConstPropAlg():
    """
    This is the constant propagation algebra, basically, it'll go 
    """

    dirty : bool = False
    g_dirty : bool = False
    tt_alg : TTypeAlg = field(default_factory=TTypeAlg)

    def _init_algebra(self, ctxt):
        ctxt.purge_keys('const') 
        self.tt_alg = TTypeAlg()
        pass

    def _init_pass(self, ctxt):
        dprint('starting cppass')
        ctxt.run_algebra(self.tt_alg)
        self.dirty = False

    def _end_pass(self, ctxt):
        self.g_dirty = self.g_dirty or self.dirty
        return self.dirty

    def _end_algebra(self, ctxt):
        return None

    def set_next(self, ident, val):
        if ((type(val) == Constant)
            and ident.has_next
            and ident['next'] != None
            and ('const' in ident['next'])):
            c = val.const_val
            inc =  ident['next']['const']
            if inc != c:
                # ignore cases where it's a boolean numeric mismatch.
                # I don't know why we keep getting 1.0 != True
                # and 0.0 != False, but we do.
                # FIXME :: Find the bug causing the above and remove the
                #         hack below. 
                print(f'{inc} != {c}')
                if (any([type(i) == bool for i in [inc,c]]) and
                    (bool(inc) == bool(c))):
                    pass
                else:
                    print(ident)
                    print(ident['next'])
                    print(ident.final)
                    raise ValueError("tried to propagate different constants.")
            
        if ident.has_next:
            self.set_next(ident['next'], val)
        else:
            if type(ident.val) != Constant:
                ident.set_next(val, color='darkgreen')
                #print(f'setting next: {ident.uniq_name} {ident.final.uniq_name}')
                if ident.final != ident:
                    self.dirty = True



    def _run(self, ident, val) -> None:
        ident = id_to_mval(ident)
  
        self.tt_alg._run(ident, ident.val) 


        if ('const' in ident and ident.has_next):
            return None
        # Just make sure we're looking only at the most recent elements 
        # ident = ident.final

        def update_exp(i):
            if i != i.final:
                self.tt_alg._run(i, i.val) 
                self._run(i.final, i.final.val)
                return i.final
            else:
                return i

        val = tmap(update_exp, ident.val)

        # If we have a scalar tensor type, and the terms are 
        if ident['tensor_type'] == TType.SCALAR:
            if type(val) == Constant:
                # If we're got a constant we've got a constant 
                ident['const'] = val.const_val
            elif type(val) in { Negate, Eq, Not, Xor, LessThan, Mag, 
                               LessThanEq, GreaterThan, GreaterThanEq,
                               Sin, Cos, Tan, Asin, Acos, Atan, Atan2}:
                # a bunch of terms should only be collapsed if all their
                # terms are constant
                if all([('const' in i) for i in tlist(val)]):
                    v = getattr(self, self.language[type(val)])(ident, val)
                    if v != None:
                        t = tmap(lambda x: x['const'], val)
                        #print(f'const propping: {type(val).__name__} from {t} to {v}')
                        
                        ident['const'] = v
                    else:
                        raise Exception("all terms are constants, should be reducible")
            elif type(val) in {IfThenElse, And, Or, Implies, Add, Mul, Pow}:
                # Others might collapse if any of their terms are constant. 
                if any([('const' in i) for i in tlist(val)]):
                    l = ident.has_next
                    getattr(self, self.language[type(val)])(ident, val)
                    #if not l and ident.has_next:
                        #print(f'const propping!: {type(val).__name__} from {val} to {ident["next"]}')
            elif type(val) in {Control, Param, Variable}:
                # others just get ignored. 
                pass
            else:
                #print(type(val).__name__)
                raise Exception("All scalar types should be accounted for.")

        # And finally, if we have given a term a new constant value, we can
        # then just that constant to the next element. 
        if 'const' in ident and type(val) != Constant:
            #dprint(f'val {ident.uniq_name} is now {ident["const"]}')
            self.set_next(ident, Constant(ident['const']))

            

    # Functions that can propagate terms if they're all constants. 

    def run_negate(self, ident, val):
        return - val.exp['const']

    def run_mag(self, ident, val):
        return math.fabs(val.exp['const'])

    def run_eq(self, ident, val):
        return bool(val.exp_a['const'] == val.exp_b['const'])

    def run_xor(self, ident, val):
        return bool(val.exp_a['const']) != bool(val.exp_b['const'])

    def run_not(self, ident, val):
        return not bool(val.exp['const'])

    def run_lessthan(self, ident, val):
        return bool(val.exp_a['const'] < val.exp_b['const'])

    def run_lessthaneq(self, ident, val):
        return bool(val.exp_a['const'] <= val.exp_b['const'])

    def run_greaterthan(self, ident, val):
        return bool(val.exp_a['const'] > val.exp_b['const'])

    def run_greaterthaneq(self, ident, val):
        return bool(val.exp_a['const'] >= val.exp_b['const'])

    def run_sin(self, ident, val):
        return math.sin(val.exp['const'])

    def run_cos(self, ident, val):
        return math.cos(val.exp['const'])

    def run_tan(self, ident, val):
        return math.tan(val.exp['const'])

    def run_asin(self, ident, val):
        return math.asin(val.exp['const'])

    def run_acos(self, ident, val):
        return math.acos(val.exp['const'])

    def run_atan(self, ident, val):
        return math.atan(val.exp['const'])

    def run_atan2(self, ident, val):
        return math.atan2(val.exp_a['const'], val.exp_b['const'])

    # Terms that shortcircuit and might not need to look at all the values
    # to propagate.

    def run_ifthenelse(self, ident, val):
        if 'const' in val.exp_cond:
            if val.exp_cond['const'] == True:
                self.set_next(ident, val.exp_true.final)
            elif val.exp_cond['const'] == False:
                self.set_next(ident, val.exp_false.final)
            else:
                raise TypeError

    def run_and(self, ident, val): 
        if ('const' in val.exp_a) and ('const' in val.exp_b):
            ident['const'] = bool(val.exp_a['const'] and val.exp_b['const'])
        elif ('const' in val.exp_a) and (val.exp_a['const'] == True):
            ident['const'] = True
        elif ('const' in val.exp_b) and (val.exp_b['const'] == True):
            ident['const'] = True
        elif ('const' in val.exp_a) and (val.exp_a['const'] == True):
            self.set_next(ident, val.exp_b)
        elif ('const' in val.exp_b) and (val.exp_b['const'] == True):
            self.set_next(ident, val.exp_a)

    def run_or(self, ident, val): 
        if ('const' in val.exp_a) and ('const' in val.exp_b):
            ident['const'] = bool(val.exp_a['const'] or val.exp_b['const'])
        elif ('const' in val.exp_a) and (val.exp_a['const'] == True):
            ident['const'] = True
        elif ('const' in val.exp_b) and (val.exp_b['const'] == True):
            ident['const'] = True
        elif ('const' in val.exp_a) and (val.exp_a['const'] == False):
            self.set_next(ident, val.exp_b)
        elif ('const' in val.exp_b) and (val.exp_b['const'] == False):
            self.set_next(ident, val.exp_a)

    def run_implies(self, ident, val): 
        if ('const' in val.exp_a) and ('const' in val.exp_b):
            ident['const'] = bool((not val.exp_a['const']) or val.exp_b['const'])
        elif ('const' in val.exp_a) and (val.exp_a['const'] == False):
            ident['const'] = True
        elif ('const' in val.exp_b) and (val.exp_b['const'] == True):
            ident['const'] = True
        elif ('const' in val.exp_a) and (val.exp_a['const'] == True):
            self.set_next(ident, val.exp_b)
        elif ('const' in val.exp_b) and (val.exp_b['const'] == False):
            self.set_next(ident, Not(val.exp_a))

    def run_add(self, ident, val):
        if ('const' in val.exp_a) and ('const' in val.exp_b):
            ident['const'] = val.exp_a['const'] + val.exp_b['const']
        elif ('const' in val.exp_a) and (val.exp_a['const'] == 0.0):
            self.set_next(ident, val.exp_b)
        elif ('const' in val.exp_b) and (val.exp_b['const'] == 0.0):
            self.set_next(ident, val.exp_a)

    def run_mul(self, ident, val):
        if ('const' in val.exp_a) and ('const' in val.exp_b):
            ident['const'] = val.exp_a['const'] + val.exp_b['const']
        elif ('const' in val.exp_a) and (val.exp_a['const'] == 0.0):
            ident['const'] = 0.0
        elif ('const' in val.exp_b) and (val.exp_b['const'] == 0.0):
            ident['const'] = 0.0
        elif ('const' in val.exp_a) and (val.exp_a['const'] == 1.0):
            self.set_next(ident, val.exp_b)
        elif ('const' in val.exp_b) and (val.exp_b['const'] == 1.0):
            self.set_next(ident, val.exp_a)

    def run_pow(self, ident, val):
        if ('const' in val.exp_a) and ('const' in val.exp_b):
            ident['const'] = math.pow(val.exp_a['const'], val.exp_b['const'])
        elif ('const' in val.exp_b) and (val.exp_b['const'] == 0.0):
            ident['const'] = 0.0
        elif ('const' in val.exp_b) and (val.exp_b['const'] == 1.0):
            self.set_next(ident, val.exp_a)
