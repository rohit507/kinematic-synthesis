from mechsynth.term import *
from typing import Generic, Tuple
from public import public

"""
This module contains the various operators that make up our (basically)
computational geometry DSL.
"""

# ## Basic Numeric Operators ## 

@public
@term
class Add(Generic[TermType]):
    """
    Standard addition, pointwise for matrices and vectors. 

    Prototypes:
      forall a \in {Number, MatMxN, Vector} => a -> a -> a 
    """
    exp_a : TermType
    exp_b : TermType

@public
@term
class Sub(Generic[TermType]):
    exp_a : TermType
    exp_b : TermType

@public
@term
class Negate(Generic[TermType]):
    """
    Multiply by -1, pointwise if needs be.

    Prototypes:
      forall a \in {Number, MatMxN, Vector} => a -> a
    """
    exp : TermType

@public
@term
class Mul(Generic[TermType]):
    """
    Multiply two numbers, pointwise for matrices and vectors.

    Prototypes:
      forall a \in {Number, MatMxN, Vector} => a -> a -> a
    """
    exp_a : TermType
    exp_b : TermType

@public
@term
class Div(Generic[TermType]):
    exp_a : TermType
    exp_b : TermType

@public
@term
class Invert(Generic[TermType]):
    """
    invert the matrix

    Prototypes:
      forall a \in {Number, MatMxN, Vector} => a -> a
    """
    exp : TermType

# ## Simple functions ##

@public
@term
class Sqrt(Generic[TermType]):
    exp : TermType

@public
@term
class Pow(Generic[TermType]):
    exp_a : TermType
    exp_b : TermType

# ## Basic Trig Functions ##

@public
@term
class Sin(Generic[TermType]):
    exp : TermType 

@public
@term
class Cos(Generic[TermType]):
    exp : TermType 

@public
@term
class Tan(Generic[TermType]):
    exp : TermType

@public
@term
class Asin(Generic[TermType]):
    exp : TermType 

@public
@term
class Acos(Generic[TermType]):
    exp : TermType 

@public
@term
class Atan(Generic[TermType]):
    exp : TermType 

@public
@term
class Atan2(Generic[TermType]):
    exp_a : TermType
    exp_b : TermType


# ## Boolean Functions ##

@public
@term
class And(Generic[TermType]):
    exp_a : TermType 
    exp_b : TermType 

@public
@term
class Or(Generic[TermType]):
    exp_a : TermType 
    exp_b : TermType 

@public
@term
class Xor(Generic[TermType]):
    exp_a : TermType 
    exp_b : TermType 

@public
@term
class Implies(Generic[TermType]):
    exp_a : TermType 
    exp_b : TermType 

@public
@term
class Iff(Generic[TermType]):
    exp_a : TermType 
    exp_b : TermType

@public
@term
class Not(Generic[TermType]):
    exp : TermType 

@public
@term
class IfThenElse(Generic[TermType]):
    exp_cond : TermType
    exp_true : TermType
    exp_false : TermType

# TODO :: OneOf(List[TermType])
@public
@term
class OneOf(Generic[TermType]):
    """
    This is basically a nested if statement.
    """
    index : TermType
    options : Tuple[TermType]

    def __tmap__(self, f):
        return OneOf(f(self.index), tuple(map(f,options)))

    def __tzip__(self, others):

        # We should never have to zip over a OneOf, they should get
        # turned into if statements at the first small step. 

        raise Exception("cannot zip over a OneOf") 

        # if any(len(self.options) != len(o.options) for o in others):
        #     raise Exception("Can't zio OneOfs, with different length lists.")

        # ts = list([self]).extend(others)

# ## Comparison Operators ##

@public
@term
class Eq(Generic[TermType]):
    exp_a : TermType
    exp_b : TermType

@public
@term
class Neq(Generic[TermType]):
    exp_a : TermType
    exp_b : TermType

@public
@term
class LessThan(Generic[TermType]):
    exp_a : TermType
    exp_b : TermType

@public
@term
class GreaterThan(Generic[TermType]):
    exp_a : TermType
    exp_b : TermType

@public
@term
class LessThanEq(Generic[TermType]):
    exp_a : TermType
    exp_b : TermType

@public
@term
class GreaterThanEq(Generic[TermType]):
    exp_a : TermType
    exp_b : TermType

@public
@term
class Between(Generic[TermType]):
    exp_val : TermType
    exp_min : TermType
    exp_max : TermType

@public
@term 
class Within(Generic[TermType]):
    exp_val  : TermType
    exp_dist : TermType
    exp_of   : TermType

# ## Vector Functions ##

@public
@term
class Dot(Generic[TermType]):
    exp_a : TermType
    exp_b : TermType

@public
@term
class Cross(Generic[TermType]):
    exp_a : TermType
    exp_b : TermType

@public
@term
class Norm(Generic[TermType]):
    exp : TermType

@public
@term
class Mag(Generic[TermType]):
    exp : TermType

@public
@term
class Dist(Generic[TermType]):
    exp_a : TermType
    exp_b : TermType

@public
@term
class Mod(Generic[TermType]):
    exp_a : TermType
    exp_b : TermType

# ## Matrix Ops ##

@public
@term
class MatMul(Generic[TermType]):
    exp_a : TermType
    exp_b : TermType

@public
@term
class Transpose(Generic[TermType]):
    exp : TermType

@public
@term
class Determinant(Generic[TermType]):
    """
    The determinant of a matrix
    """
    exp : TermType

@public
@term
class CrossMat(Generic[TermType]):
    """
    The cross product matrix for a vector. i.e. `CrossMat(a)` is the matrix
    such that for all `b`, `CrossMat(a) @ b == Cross(a,b)`
    """
    exp : TermType
