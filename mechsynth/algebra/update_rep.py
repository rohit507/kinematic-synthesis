from mechsynth.term import *
from mechsynth.symbolic.value import *
from mechsynth.symbolic.geom import *
from mechsynth.symbolic.object import *
from mechsynth.symbolic.relation import *
from mechsynth.symbolic.model import *
from mechsynth.symbolic.model_ops import *
from pytest import * # type: ignore

"""
This is a simple pass, whose job is to basically garbage collect the terms
in a context and flatten chains of replacement relationships. 
"""
