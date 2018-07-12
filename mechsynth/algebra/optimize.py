from mechsynth.term import *
from mechsynth.symbolic.value import *
# from mechsynth.symbolic.geom import *
# from mechsynth.symbolic.object import *
# from mechsynth.symbolic.relation import *
from mechsynth.symbolic.model import *
from enum import Flag, auto

from mechsynth.algebra.small_step import *
from mechsynth.algebra.util import *
from mechsynth.algebra.simplify import *
from mechsynth.algebra.sympy import *

from pprint import *
from tabulate import tabulate

"""
This is the handler that takes care of running a single optimization pass,
as well some stuff that checks correctness.
"""

def dprint(*args, **kwargs):
    if True:
        print(*args, **kwargs) 

def model_stats(model, pass_name):
    return {
        'pass_name' : pass_name, 
        '# terms'   : len(model._id_map),
        '# frames'  : len(model.ref_frames),
        '# objs'    : len(model.objects),
        '# params'  : len(model.parameters),
        '# controls': len(model.controls),
        '# vars'    : len(model.variables),
        '# asserts' : len(model.assertions),
        '# assums'  : len(model.assumptions),
        '# consts'  : len(model.constraints),
        '# reqs'    : len(model.guarantees),
        '# costs'   : len(model.cost_terms),}

@model_algebra
class OptimizeAlg():
    """
    This runs all the different algebras we've defined once.
    Producing a new model at the end. 
    """

    name    : str = ""
    counter : int = 1
    stats   : list = []

    dirty : bool = False
    num_steps : int = 3
    

    _small_step : SmallStepAlg = field(repr=False)
    _eq_elim    : EqElimAlg    = field(repr=False)
    _const_prop : ConstPropAlg = field(repr=False)
    _sympy      : SympyAlg     = field(repr=False)
    _prune      : PruneAlg     = field(repr=False) 

    _no_run : bool = True 


    def __init__(self, name, counter, stats=None, num_steps=3):
        self.name = name
        self.counter = counter
        self.stats = stats
        self.num_steps = num_steps

    def _init_algebra(self, ctxt):

        self.dirty = False

        self._small_step = SmallStepAlg(max_runs = self.num_steps)
        self._eq_elim = EqElimAlg()
        self._const_prop = ConstPropAlg()
        self._sympy = SympyAlg()
        self._prune = PruneAlg(f'{self.name}_{self.counter}', keep_relevant=True) 

        self._append_stats(ctxt,'initial')

        dprint(f'{self.name} ({self.counter}): Running Small Step Reduction')
        ctxt.run_algebra(self._small_step)
        self._append_stats(ctxt,'post_small_step')

        dprint(f'{self.name} ({self.counter}): Running Equality Elimination Pass')
        ctxt.run_algebra(self._eq_elim)
        self._append_stats(ctxt,'post_eq_elim')

        dprint(f'{self.name} ({self.counter}): Running Constant Propagation Pass')
        ctxt.run_algebra(self._const_prop)
        self._append_stats(ctxt,'post_const_prop')


        print_graphs(f'opt_{self.name}_pre_sp({self.counter})', ctxt)

        dprint(f'{self.name} ({self.counter}): Running Sympy Pass')
        ctxt.run_algebra(self._sympy)
        self._append_stats(ctxt,'post_sympy')

    def _append_stats(self, ctxt, stage): 
        if self.stats != None:
            self.stats.append(
                model_stats(ctxt, f'{self.name}_{self.counter}_{stage}'))

    def _init_pass(self, ctxt):
        pass

    def _end_pass(self, ctxt):
        return False

    def _end_algebra(self, ctxt):
        """
        prunes and propagates the dirty bits
        """

        print_graphs(f'opt_{self.name}_pre({self.counter})', ctxt)

        dprint(f'{self.name} ({self.counter}): Pruning Graph')
        new_ctxt = ctxt.run_algebra(self._prune)
        self._append_stats(ctxt,'post_prune')

        self.dirty = (self._small_step.dirty
                      or self._eq_elim.dirty
                      or self._const_prop.g_dirty
                      or self._prune.dirty)

        return (self.dirty, new_ctxt) 

    def _run(self, ident, val) -> None:
        raise NotImplementedError

def print_stats(name, stats):
    pathlib.Path('__debug/').mkdir(parents=True, exist_ok=True)
    f = open(f'__debug/{name}_stats.txt',"w")
    f.write(tabulate(stats, headers='keys'))
    f.close()
