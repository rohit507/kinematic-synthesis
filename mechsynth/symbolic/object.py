from typing import *
from mechsynth.term import *           # type: ignore 
from mechsynth.symbolic.value import * # type: ignore 
from mechsynth.symbolic.geom import *  # type: ignore
from dataclasses import *
from abc import *
from public import public 

"""
This module contains objects that exist within our kinematic model. These are
the elements that will eventually be turned into actual, physical,
bits-and-bobs over the course of the synthesis process.
"""


@public
@term
class Object(Generic[TermType],ABC):
    """
    An object may or may not exist.
    If it does, its constraints apply, of it does not, they don't. 
    """
    exists : TermType

@public
@term
class Body(Frame[TermType],Object[TermType]):
    """
    For our purposes, a rigid body is pretty identical to a reference frame.
    It is a set of points that are all related, by some transform in SE, to
    the default reference frame.
    """

@public
@term
class HasBody(HasFrame[TermType]):
    """
    Basically, require a reference to the parent body instead of the parent
    reference frame.
    """

@public
@term
class Anchor(Point,HasBody[TermType], Object[TermType]):
    """
    An anchor is a point that is fixed relative to its given frame.
    All of its coordinates must be parameters or constants.

    TODO :: This has no notion of direction or rotation, so the system is
           free to make it point in any direction. There should be other
           objects that are not subject to this. 
    """
    direction : TermType
    handle : TermType
    pass

# TODO :: Arrow (an anchor with a chosen direction, rotationally symmetric)
# TODO :: Handle (an arrow with a chosen perpendicular vector, no symmetries)

# @termnchor
# class Joint(FrameTransform[TermType],ABC):
#     """
#     The abstract class for any connection between two rigid bodies that can
#     be turned into an actual physical thing.
#     """

# TODO :: Consider call these revolute and prismatic joints.
@public
@term
class Hinge(FrameTransform[TermType], Object[TermType]):
    """
    A hinge is a joint between two bodies that allow them to rotate relative
    to each other along a fixed axis. 

    This is going to be a transformation between two frames, with a known
    line of coincidence, and angle of rotation.

    At `t = 0` the angle must be equal to 0.

    NOTE :: 0 is always in the acceptable range of angles here.
dent
               0 < max_bound < 2*pi
           -2*pi < min_bound < 0
       min_bound < angle_var < max_bound 


    TODO :: Don't forget to handle the bounds on the angle of rotation, it's
           either a full 360 degree+ rotation, or some fixed set of sub-angles.
    """
    axis : TermType
    angle : TermType
    min_angle : TermType
    max_angle : TermType

@public
@term
class Slide(FrameTransform[TermType], Object[TermType]):
    """
    This is a sliding joint that allows two bodies to slide past each other
    along a shared axis.

    NOTE :: 0 is always in the acceptable range of distances here.

               0 < max_bound  < +limit
          -limit < min_bound  < 0
       min_bound < offset_var < max_bound 

    TODO :: Rationalizing this will be interesting, esp if we have some notion
           of error. 
    """
    axis : TermType
    offset : TermType
    min_offset : TermType
    max_offset : TermType

# TODO :: Joints that are uncorrelated combinations of these (i.e. just what you
#        get when you compose them and factor out the missing internal reference
#        frame.
#
#        x^2 
#         
#        Hinge * Hinge = 2-DOF hinge / ball-socket joint
#        Slide * Slide = planar slide (move \in 2D)
#        Hinge * Slide = rotating rod mechanism (if shared axes)  

# TODO :: Joints that are correlated compositions of these, where both the
#        degrees of freedom are tied by an internal ratio
#
#        x^2
#
#        Hinge * Hinge = Gear (non-shared axes) 
#        Hinge * Slide = Rack and Pinion (orthogonal axes) / Screw (shared axes)
#        Slide * Slide = ... well it's a slide at a different angle :V 

# NOTE :: Huh, all of this must be stuff that engineers and mathematicians,
#        discovered ages and ages ago. It's cool to be learning/recreating it
#        for myself. 
