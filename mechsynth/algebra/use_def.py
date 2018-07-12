from mechsynth.term import *
from mechsynth.symbolic.value import *
# from mechsynth.symbolic.geom import *
# from mechsynth.symbolic.object import *
# from mechsynth.symbolic.relation import *
from mechsynth.symbolic.model import *
from enum import Flag, auto

"""
This is just a standard use def analysis pass that helps us track which
terms we can keep track of and which we can throw away.
"""

class UseType(Flag):
    CONST = 0
    PARAM = auto()
    CONTROL = auto()
    VAR = auto()

@model_algebra
class UseDefAlg():
    """
    We just run this algebra to gather sets of parameters, controls, and
    variables that are either *used by* or *defined over* each term.


    """
    def _init_algebra(self, ctxt):
        ctxt.purge_keys({'uses', # The list of params, controls and vars that
                                 # this term uses
                         'defines', # The list of terms that this term is used
                                    # in defining
                         'use_type' # ode of the follow
        })

    def _init_pass(self, ctxt):
        pass

    def _end_pass(self, ctxt):
        return False

    def _end_algebra(self, ctxt):
        return None

    def _run(self, ident, val) -> None:
        """
        We override the run function because it's basically the same code
        except for the three core classes.
        """

        uses = set()
        use_type = UseType.CONST

        # If we're something interesting, we use ourselves.
        if isinstance(val, Param):
            uses.add(ident)
            #use_type |= UseType.PARAM
        elif isinstance(val, Control):
            uses.add(ident)
            #use_type |= UseType.CONTROL
        elif isinstance(val, Variable):
            uses.add(ident)
            #use_type |= UseType.VAR

        def add_used(i):
            uses.update(i['uses'])
            #nonlocal use_type
            #use_type |= i['use_type']
            

        # Add any terms our children use
        tmap(add_used, val)

        def add_defines(i):
            i['defines'].add(ident)
            tmap(add_defines,i.val)

        #tmap(add_defines, val)

        ident['uses'] = uses
        #ident['defines'] = set()
        #ident['use_type'] = use_type
