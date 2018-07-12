from mechsynth.term import *
from typing import Generic
from public import public

"""
This module captures relations between terms, generally these act as
informative flags we use during various passes. 
"""

# TODO :: Decide if this is necessary at all or would be better done in the
#        KV-Store
#
# @public
# @term
# class AtInit(Term[TermType]):
#     """
#     The value of a term at `t = 0`, as we run through the network adding
#     various simplifications and notions of equality, this lets us say
# 
#     "this value, but at time `t == 0`" 
#     """
#     exp : TermType 

@public
@term
class GetMember(Generic[TermType]):
    """
    The operation for getting a sub-member/child elem of value. Different
    values have different types of members.
    """
    member : str
    exp : TermType

@public
@term
class AtInitial(Generic[TermType]):
    """
    The value at some initial condition.
    """
    exp : TermType
