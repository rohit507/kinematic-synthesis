from typing import * # type: ignore
from mechsynth.term import * # type: ignore
from dataclasses import *
from abc import *
from enum import Flag, auto
from public import public

"""
This module defines basic values as terms that we can store information in.
These will eventually be wrapped by complex elements. For the moment though,
they can be used to store things.

In other modules we will wrap them in more descriptive types that actually
represent things in the kinematic systems we model.

TODO :: Basic operator passthrough support for these things. This is going to
       be (very) hacky, but will probably make our lives a bit easier.

"""

@public
class ValType(Flag):
    """
    The three core types of numeral we use within this system, these directly
    translate into types within sympy, dreal, and other interfaces.

    We use the 'Flag' class here instead of 'Enum' because it gives us a
    convenient representation of type ambiguity. Just '&' two types together
    to get the most precise one that works, and failure are just false. 

    Members:
      REAL := Real numbers, usually just floats 
      INT  := Integers, these should be limited to parameters
      BOOL := Booleans, these should be limited to parameters

      ANY  := Any of the above
    """
    REAL = auto()
    INT  = auto()
    BOOL = auto()
    ANY = ~0

@public
class ValUnit(Flag):
    """
    These are semantic units meant for debug, and an internal type checking
    pass.
    
    We use 'Flag' instead of 'Enum' here for the same reasons as ValType. 

    Members:
      DISTANCE := Distance in some unit 'D'
      VELOCITY := Speed in 'D / sec'
      ANGLE    := In radians, limited to 
      ROTATION := In radians, but can go outside of the +/- Pi range, each
                  additional 2*Pi represents another entire circle away from
                  the base state.
      ANGULAR_VELOCITY := radians / sec
      UNITLESS := Many ratios, and matrix elements are effectively unitless
                  and we gain nothing from trying to keep track of them.

      ANY := Any of the above
    """
    DISTANCE = auto()
    VELOCITY = auto()
    ANGLE    = auto()
    ROTATION = auto()
    ANGULAR_VELOCITY = auto()
    UNITLESS = auto()
    ANY = ~0

@public
@term
class Value(Generic[TermType]):
    """
    A value that can store a primitive real number, integer, or boolean. 
    """

@public
@term
class Param(Generic[TermType]):
    """
    An unknown parameter value, that is fixed for all `t`
    """
    name : str
    v_type : ValType


@public
@term
class Constant(Generic[TermType]):
    """
    A constant term, with a fixed and known vald type.
    """
    const_val : Union[float,int,bool]

@public
@term
class Control(Generic[TermType]):
    """
    A user-driven control input, with an initial condition for `t == 0`.
    The initial condition can be either a constant or parameter.

    For now, it is always a real number, that may change in the future. 
    """
    name : str
    initial_condition : TermType

@public
@term
class Variable(Generic[TermType]):
    """
    This is a variable within the kinematic system that, given some set of
    parameters and controls, satisfies the constraints in the model.

    In a fully finished model there should only be only one possible
    assignment of each variable given some parameters.

    This is the same as a degree-of-freedom, a dependent variable that we
    would like to take on a single fixed value at any point.

    TODO :: Figure out whether variables also need an explicit initial
           condition. If they are fixed functions, then we're fine, but
           an explicit initial condition might help a lot with the solving
           process, since there's a ton of places where we can assume the
           initial condition is 0, and that must be w/in the bounds. 
    """
    name : str
    initial_condition : TermType

# TODO :: Constraint
#        A list(set?) of constraints that are equivalent to the above.

@public
@term
class Mat3x1(Generic[TermType]):
    """
    A 3 element column vector. We're avoiding adding too many things to the
    namespace, because we want to define both points and vectors in spatial
    terms.

    This is why we're using `Mat` as a type prefix for all of these elements. 
    """
    x : TermType
    y : TermType
    z : TermType

@public
@term
class Mat4x1(Generic[TermType]):
    """
    A 4 element column vector. 
    """
    x : TermType
    y : TermType
    z : TermType
    w : TermType

@public
@term
class Mat3x3(Generic[TermType]):
    """
    A 3x3 matrix made up of 3 column vectors.
    """

    c1 : TermType
    c2 : TermType
    c3 : TermType


@public
@term
class Mat4x4(Generic[TermType]):
    """
    A 4x4 matrix made up of 4 column vectors. 
    """

    c1 : TermType
    c2 : TermType
    c3 : TermType
    c4 : TermType
