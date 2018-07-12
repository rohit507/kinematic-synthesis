from typing import *
from mechsynth.term import * # type: ignore
from mechsynth.symbolic.value import * # type: ignore
from dataclasses import *
from abc import *
from enum import Enum

"""
This module contains geometric constructs that have a spatial or kinematic
meaning. 

Often these are just going to have a slightly nicer wrapper type around a
value for the purposes of type checking, but some are going to be more complex.
"""


@term
class HasFrame(Term[TermType],ABC):
    """
    Elements that inherit from this have a frame of reference in which they
    make sense, and trying to use those numbers in a different frame will
    lead to errors.
    """
    parent_frame : TermType

@term
class Point(Mat4x1, HasFrame[TermType]):
    """
    A point in R^3 represented in homogenous coordinates as a column vector of
    transpose([x , y, z, 1]), within a particular reference frame.

    tThese are a position within the space, and are transformed as such when
    converted from one space to another, both rotated as translated.
    """
    pass

@term
class Vector(Mat4x1, HasFrame[TermType]):
    """
    A vector in R^3 represented in homogenous coordinates as a column vector
    transpose([x, y, z, 0]), along with a reference to its parent reference
    frame.

    These are directions in a given reference frame, and will only be rotated
    when moved into a new reference frame. 
    """
    pass

@term
class Unit_Vector(Vector[TermType]):
    """
    A vector of magnitude 1, very important for a number things, so we're
    giving it its own type. Yay!
    """
    pass

@term
class SOMat3x3(Mat3x3[TermType]):
    """
    A special orthogonal matrix represents the group of rotations on R^3.
    In particular the set of columns of this matrix (x, y, z) are the
    three basis vectors of the newly rotated space, within the default
    basis (ID).

    The special orthogonal group imposes the following constraints over
    x, y, and z:

      1 = ||x|| = ||y|| = ||z|| 
      0 = dot(x,y) = dot(x,z) = dot(y,z) 
      z = cross(x,y)

    (we're assuming this is made of vectors, just for convenience)

    """
    pass

@term
class SEMat4x4(Mat4x4[TermType]):
    """
    A special euclidean matrix in R^3, these matrices are transformations from
    coordinate frame `b` to `a`. They can be assembled by taking a special
    orthogonal matrix representing a rotation (named `R_{a,b}`), and the
    coordinates of `b`s origin in frame `a` (called `p_{a,b}`) as follows:

      |R_{a,b} , p_{a,b}|
      |[0,0,0] ,   1    |

    The usual shorthand is `q_a = (g_{a,b} : SEMat4x4) @ q_b` where `q` is a
    vector or point in the respective frame. 
    """

@term
class Frame(SEMat4x4[TermType]):
    """
    A reference frame represented as a transformation matrix from itself
    to the default deference 

    i.e.
      self @ ((Vector|Point) \in self) == (Vector|Point \in Default)

    The other key property of a Frame is that every frame **MUST** at `t = 0`
    be identical to the default frame. This lets us have sane descriptions of
    systems, because we can rely on all the reference frames to overlap at
    "description time".

    Seriously, that latter property is *why* this is a special entry in the
    class hierarchy, rather than just using an SEMat4x4 directly. 
    """
    pass

@term
class FrameTransform(SEMat4x4[TermType], HasFrame[TermType]):
    """
    A transformation from one reference frame to another, this just
    keeps tracks of which frame goes where so that we can catch stupid
    mistakes and get some nice compositionality properties.

    Each frame transform also has the following:
      axis := The line about which this transform shifts
      offset := the distance along the line we move
      angle := the rotation along the line
    """
    target_frame   : TermType

@term
class InFrame(Generic[TermType]):
    """
    Given an object that is within a frame, get a representation/position
    with respect to some other frame.
    """
    obj : TermType
    target_frame : TermType

@term
class Line(HasFrame[TermType]):
    """
    A line in a given frame can be represented as a set of 6 terms with a
    handful of constraints, known as its plucker coordinates.

    Basically, given:
      p  :: point on the line
      q  :: direction vector of the line (unit-vector)
      q0 := cross(p,q) :: The moment vector 

    The plucker coordinates are the pair (q, q_0), with the following
    extra properties:

      - dot(q, q0) = 0
      - Lines are same if multiplied by any constant:

          (q, q0) === (k * q, k * q0)

      - q0 is normal to the plane containing the origin and the line.
      - |q0|/|q| is dist from origin to the line.
      - If q0 == 0, then the line passes through the origin.
      - Point x is on line if:

          cross(x,q) == q0

      - Point closest to origin is:

          cross(dir,moment)/dot(dir,dir)

      - Convert to new frame, just multiply both q,q0 by matrix, both vectors.

    NOTE :: Because we ignore lines at infinity we can normalize the coordinate
           axis with respect to `q` and use strict equality in the solver. 

    Cite:

      - http://orb.olin.edu/plucker.pdf
      - http://www.cs.rpi.edu/~trink/Courses/RobotManipulation/lectures/lecture8.pdf

    TODO :: https://link.springer.com/content/pdf/10.1007/978-3-319-11550-4_5.pdf
           Might be better way to represent lines and planes in 3d space?
           The <unit-vector-in-dir, point-closest-to-origin> formulation is
           pretty cool, and this has a nice planar representation too. 
           At the very least it might be a better source for things.
    """

    direction : TermType
    moment : TermType

    @property
    def __attrs__(self): 
        hints = None
        if not self.__class__.__name__ in getattr(Line,'__hints'): 
            hints = dict()
            for t in self.__class__.mro():
                hints.update(get_type_hints(t))
            getattr(Line,'__hints')[self.__class__.__name__] = hints
        else:
            hints = getattr(Line,'__hints')[self.__class__.__name__]

        return hints

    def __tmap__(self, f):

      # We must get all the fields in the term including those inherited from
      # parent classes. 
      hints = dict()
      for t in self.__class__.mro():
        hints.update(get_type_hints(t))

      kargs = dict()               # Key-args we assemble

      # If the argument has the correct type variable, apply our
      # function to it, and store it in the replaced kargs
      for arg in hints:
        if hints[arg] == TermType:
          kargs[arg] = f(getattr(self,arg))
        else:
          kargs[arg] = getattr(self,arg)

      # Call the initializer function with the new values.
      return self.__class__(**kargs)

    def __tzip__(self, others):

      # We must get all the fields in the term including those inherited from
      # parent classes. 
      hints = dict()
      for t in self.__class__.mro():
        hints.update(get_type_hints(t))

        kargs = dict()               # Key-args we assemble

      ts = list([self])

      for o in others:
        # FIXME :: Why does this work when .extend() doesn't?
        ts.append(o)

      if any(type(self) != type(o) for o in others):
        raise Exception("All zipped elements must be of the same type.")

      # If the argument has the correct type variable, apply our
      # function to it, and store it in the replaced kargs 
      for arg in hints:
        if hints[arg] == TermType:
          kargs[arg] = tuple([getattr(t,arg) for t in ts])
        else:
          kargs[arg] = getattr(self,arg)
          if any(getattr(self,arg) != getattr(o,arg) for o in others):
            raise Exception("Other elements of tuple don't properly match.")

      # Call the initializer function with the new values.
      return self.__class__(**kargs)

# TODO :: Planes and other objects will probably end up being important come
#        time to implement bounding boxes, speed, and error bars. 
