from typing import *
from mechsynth.context           import * # type: ignore
from mechsynth.symbolic.value    import * # type: ignore
from mechsynth.symbolic.geom     import * # type: ignore 
from mechsynth.symbolic.object   import * # type: ignore
from mechsynth.symbolic.operator import * # type: ignore
from mechsynth.symbolic.relation import * # type: ignore
from mechsynth.symbolic.model    import * # type: ignore
from mechsynth.errors            import * # type: ignore
from dataclasses import *
import itertools
import inspect
import wrapt
from enum import Enum, auto
from contextlib import contextmanager
import contextvars as cv
from infix import shift_infix as infix
from public import public

"""
This module contains a lot of utility functions that can be used when in a
build environment of a model.

these are wrapped so that we don't have to
"""

def tup_map(f, *args):
    return tuple(map(f,args)) if len(args) > 1 else tuple(map(f,args.pop()))

@public
@within_model_context
def soft_insert(model, val):
    if isinstance(val, ID):
        return val
    else:
        return model.insert(val)

@public
@within_model_context
def const(model: Model, val : Union[float,int,bool], type_hint=None) -> MVal[Constant]:
    """
    Create a constant value within this model.
    """
    
    if not any([type_hint in {n,None} and type(val) == n for n in {bool, float, int}]):
        raise TypeError

    return model._const(val)

@public
@within_model_context
def parameter(model : Model,
          name : str,
          type_hint : Union[Type, ValType],
          max_bound : Union[None, float] = None,
          min_bound : Union[None, float] = None, 
          provenance : Union[None, MVal, Set[MVal]] = None) -> MVal[Param]:
    """
    Create a new parameter with some given name and type.

    params:
      name := Human readable name
      type_hint := either 'bool', 'float', or 'int'. Used to determine the
                  type of the parameter. 
    """
    return model._param(name,
                        type_hint,
                        max_bound,
                        min_bound,
                        provenance)


@public
@within_model_context
def control(model : Model,
            name : str,
            initial_condition : Union[None, float, MVal] = None,
            max_bound : Union[None, float, MVal] = None,
            min_bound : Union[None, float, MVal] = None, 
            provenance : Union[None, MVal, Set[MVal]] = None
            ) -> MVal[Control]:
    """
    Create a new control variable. Will not collide with other control
    parameters even if initial-conditions or name are the same. 

    These should pretty much always be made by hand.

    TODO :: Expand so it can do things like add points, lines, etc... as controls

    Params:
      name := Human readable name for this parameter.
      initial_condition := Creates a new parameter, if not constant or
                           parameter is given.
      provenance := as usual. 
    """
    return model._control(name,
                          initial_condition,
                          max_bound,
                          min_bound,
                          provenance)

@public
@within_model_context
def var(model : Model,
        name : str,
        initial_condition : Union[None, float, MVal] = None,
        max_bound : Union[None, float, MVal] = None,
        min_bound : Union[None, float, MVal] = None, 
        provenance : Union[None, MVal, Set[MVal]] = None) -> MVal[Control]:
    """
    Create a new variable.

    Params:
      name := Human readable name for this parameter.
      initial_condition := Creates a new parameter, if not constant or
                           parameter is given.
      provenance := as usual. 
    """
    
    return model._variable(name,
                           initial_condition,
                           max_bound,
                           min_bound,
                           provenance)

def _mk_key_val(model, prefix, inserter, *args, name=None):
    i = 1
    for val in args:
        t_name = f'{prefix}({name})'
        if (name != None) and len(args) > 1:
            t_name = f'{t_name}-{i}'
        elif name != None:
            pass
        else:
            t_name = None
        inserter(val, t_name)
        i += 1

@public
@within_model_context
def m_assert(model, *args, name=None)-> None:
    _mk_key_val(model, 'assert', model.add_assertion, *args, name=name) 

@public
@within_model_context
def assume(model, *args, name=None) -> None:
    _mk_key_val(model, 'assume', model.add_assumption, *args, name=name) 

@public
@within_model_context
def constrain(model, *args, name=None) -> None:
    _mk_key_val(model, 'constraint', model.add_constraint, *args, name=name) 

@public
@within_model_context
def require(model, *args, name=None) -> None:
    _mk_key_val(model, 'require', model.add_guarantee, *args, name=name) 
    

@public
@within_model_context
def penalize(model, *args, name=None) -> None:
    _mk_key_val(model, 'penalize', model.add_cost, *args, name=name) 

@public
@within_model_context
def add(model : Model, a, b):
    return a + b

@public
@within_model_context
def sub(model : Model, a, b):
    return a - b

@public
@within_model_context
def negate(model : Model, a):
    return a.__negate__()

@public
@within_model_context
def mul(model : Model, a, b):
    return a * b

@public
@within_model_context
def div(model : Model, a, b):
    return a / b

@public
@within_model_context
def invert(model : Model, a):
    return a.__invert__()

@public
@within_model_context
def sqrt(model : Model, a):
    return a ** 0.5

@public
@within_model_context
def pow(model : Model, a, b):
    return a ** b

@public
@within_model_context
def square(model : Model, a):
    return a ** 2.0

@public
@within_model_context
def sin(model : Model, a):
    return model.insert(Sin(a))

@public
@within_model_context
def cos(model : Model, a):
    return model.insert(Cos(a))

@public
@within_model_context
def tan(model : Model, a):
    return model.insert(Tan(a))

@public
@within_model_context
def asin(model : Model, a):
    return model.insert(Asin(a))

@public
@within_model_context
def acos(model : Model, a):
    return model.insert(Acos(a))

@public
@within_model_context
def atan(model : Model, a):
    return model.insert(Atan(a))

@public
@within_model_context
def atan2(model : Model, a,b):
    return model.insert(Atan2(a,b))

@public
@within_model_context
def oneof(model : Model, *args):
    # Gets the arguments, creates the integral parameter,
    # and returns the oneof object ...
    return NotImplementedError

@public
@within_model_context
def m_and(model : Model, a,b):
    return model.insert(And(a,b))

@public
@within_model_context
def m_or(model : Model, a,b):
    return model.insert(Or(a,b))

@public
@within_model_context
def m_not(model : Model, a):
    return model.insert(Not(a))

@public
@within_model_context
def m_xor(model : Model, a,b):
    return model.insert(Xor(a,b))

@public
@within_model_context
def m_implies(model : Model, a,b):
    return model.insert(Implies(a,b))

@public
@within_model_context
def m_iff(model : Model, a,b):
    return model.insert(Iff(a,b))

@public
@within_model_context
def m_not(model : Model, a):
    return model.insert(Not(a))

@public
@within_model_context
def if_then_else(model : Model, c, t,f):
    return model.insert(IfThenElse(c,t,f))

@public
@within_model_context
def eq(model : Model, a,b):
    return model.insert(Eq(a,b))

@public
@within_model_context
def neq(model : Model, a,b):
    return model.insert(Neq(a,b))

@public
@within_model_context
def less_than(model : Model, a,b):
    return model.insert(LessThan(a,b))

@public
@within_model_context
def less_than_eq(model : Model, a,b):
    return model.insert(LessThanEq(a,b))

@public
@within_model_context
def greater_than(model : Model, a,b):
    return model.insert(GreaterThan(a,b))

@public
@within_model_context
def greater_than_eq(model : Model, a,b):
    return model.insert(GreaterThanEq(a,b))

@public
@within_model_context
def dot(model : Model, a,b):
    return model.insert(Dot(a,b))

@public
@within_model_context
def cross(model : Model, a,b):
    return model.insert(Cross(a,b))

@public
@within_model_context
def norm(model : Model, a):
    return model.insert(Norm(a))

@public
@within_model_context
def mag(model : Model, a):
    return model.insert(Mag(a))

@public
@within_model_context
def length(model : Model, a):
    return mag(a)

@public
@within_model_context
def matmul(model : Model, a, b):
    return model.insert(MatMul(a,b))

@public
@within_model_context
def determinant(model : Model, a):
    return model.insert(Determinant(a))

@public
@within_model_context
def crossmat(model : Model, a):
    return model.insert(CrossMat(a))

@public
@within_model_context
def mat3x1(model : Model, x, y, z):
    return model.insert(Mat3x1(x,y,z))

@public
@within_model_context
def mat4x1(model : Model, x, y, z, w):
    return model.insert(Mat4x1(x,y,z,w))

@public
@within_model_context
def mat3x3(model : Model, c1, c2, c3):
    return model.insert(Mat3x3(c1,c2,c3))

@public
@within_model_context
def mat4x4(model : Model, c1, c2, c3, c4):
    return model.insert(Mat4x4(c1,c2,c3,c4))

@within_model_context
def extendmat(model, m3):
    c1 = mat4x1(m3.c1.x, m3.c1.y, m3.c1.z, 0.0)
    c2 = mat4x1(m3.c2.x, m3.c2.y, m3.c2.z, 0.0)
    c3 = mat4x1(m3.c3.x, m3.c3.y, m3.c3.z, 0.0)
    c4 = mat4x1(0.0, 0.0, 0.0, 1.0)
    return mat4x4(c1, c2, c3, c4)

@public
@within_model_context
def identity(model : Model, a):
    if a == 3:
        return mat3x3(c1=mat3x1(1.0,0.0,0.0),
                      c2=mat3x1(0.0,1.0,0.0),
                      c3=mat3x1(0.0,0.0,1.0))
    elif a == 4:
        return mat4x4(c1=mat4x1(1.0,0.0,0.0,0.0),
                      c2=mat4x1(0.0,1.0,0.0,0.0),
                      c3=mat4x1(0.0,0.0,1.0,0.0),
                      c4=mat4x1(0.0,0.0,0.0,1.0))
    else:
        raise Exception("Identity element only exists for 3x3 and 4x4 matrices.")

@within_model_frame_context
def fixed_var(model : Model, frame : Frame, name,val):
    """
    Create a fixed variable with some name from a value that
    could be a constant float, an mval, or None.

    Generates a constant if there's a fixed input.
    """

    if isinstance(val, float):
        return const(val)
    elif isinstance(val, ID):
        val = id_to_mval(val)
        val.add_name(name, is_rel = True)
        return val
    elif val == None:
        return parameter(name, float)
    else:
        raise Exception("unsupported input type")

@within_model_frame_context
def mobile_var(model : Model, frame : Frame, name,val):
    """
    create a non-fixed variable from a name and value. 
    """

    if isinstance(val, float):
        return var(name, initial_condition = val)
    elif isinstance(val, ID):
        val = id_to_mval(val)
        val.add_name(name, is_rel = True)
        return val
    elif val == None:
        return var(name)
    else:
        raise Exception("unsupported input type")

@within_model_context
def assert_frames_equal(model, name, fa, fb):
    """
    We need to make sure that the frame we are pulling our value from
    is the same as the frame we're supposed to be writing to. This
    is going to be one of those annoying things that we should resolve
    with constant propagation, and when we get an error we can trace the
    transformations that got us there. (hopefully, such error management) 
    """
    fa, fb = tup_map(id_to_mval,fa,fb)
    if fa != fb :
        m_assert(fa.equals(fb), name=f'frame_equality({name})')

@within_model_frame_context
def make_xyz(model : Model, frame : Frame,
             name : str,
             val : Union[None, MVal] = None, 
             x : Union[None, float, MVal] = None,
             y : Union[None, float, MVal] = None,
             z : Union[None, float, MVal] = None,
             fixed : bool = False) -> MVal[Point]:


    make_var = fixed_var if fixed else mobile_var

    nums = [(x,'x'), (y,'y'), (z,'z')]
    out = list()

    for num, n_name in nums:
        v = None
        sub_name = name + '.' + n_name

        if val == None:
        # If we're not given a value then we can just default to the
        # sub element behavior.
            v = make_var(sub_name, num)
        elif isinstance(val, Term):
            v = getattr(val,n_name) 
            v.add_name(sub_name, is_rel=True)
        elif isinstance(val, ID):

            # If we are given a value, we treat this like a cast of sorts,
            # grab those elements we need to from the value and move on. 
            val = id_to_mval(val)
            v = getattr(val,n_name) 
            v.add_name(sub_name, is_rel=True)
        else:        
            raise Exception("unsupported input type")

        out.append(v)

    return tuple(out)
# Add point (x,y,z default to params, in frame)
@public
@within_model_frame_context
def point(model : Model, frame : Frame,
          name : str,
          val : Union[None, MVal] = None, 
          x : Union[None, float, MVal] = None,
          y : Union[None, float, MVal] = None,
          z : Union[None, float, MVal] = None,
          fixed : bool = False) -> MVal[Point]:
    """
    creates a point in some frame.

    params:
      model := our parent model
      frame := the frame in which this point exists.
      name := The human readable name of this point
      val := if exists, will take the x, y, and z terms of whatever is put here
             and use those as the seed for this point. 
      x := the x value
      y := the y value
      z := the z value
      fixed := if true, initializes points as parameters/constants,
               otherwise variables/initial conditions. 
    """
    x, y, z = make_xyz(name, val, x, y, z, fixed)
    w = const(1.0) 

    if isinstance(val, ID) and hasattr(val.val, 'parent_frame'):
        assert_frames_equal(name, val.frame, frame)

    return model.insert(Point(parent_frame=frame, x=x,y=y,z=z,w=w), name)


@public
@within_model_frame_context
def origin(model : Model, frame : Frame):
    """
    Gives you the origin in some frame, generally the default if unspecified. 
    """
    return point('origin',x=0.0, y=0.0, z=0.0, fixed = True)

@public
@within_model_frame_context
def vector(model : Model, frame : Frame, name : str,
           val : Union[None, MVal] = None, 
           x : Union[None, float, MVal] = None,
           y : Union[None, float, MVal] = None,
           z : Union[None, float, MVal] = None,
           fixed : bool = False) -> MVal[Point]:
           
    """
    creates a vector in some frame.

    params:
      model := our parent model
      frame := the frame in which this vector exists.
      name := The human readable name of this point
      val := if exists, will take the x, y, and z terms of whatever is put here
             and use those as the seed for this vector. 
      x := the x value
      y := the y value
      z := the z value
      fixed := if true, initializes indices as parameters/constants,
               otherwise variables/initial conditions. 
    """

    x, y, z = make_xyz(name, val, x, y, z, fixed)
    w = const(0.0) 

    if isinstance(val, ID):
        assert_frames_equal(name, val.frame, frame)

    return model.insert(Vector(x=x,y=y,z=z,w=w,parent_frame=frame), name)


@public
@within_model_frame_context
def unit_vector(model : Model, frame : Frame, name : str,
                val : Union[None, MVal] = None, 
                x : Union[None, float, MVal] = None,
                y : Union[None, float, MVal] = None,
                z : Union[None, float, MVal] = None,
                fixed : bool = False,
                make_unit : bool = False
                ) -> MVal[Point]:     
    """
    creates a unit_vector in some frame.

    params:
      model := our parent model
      frame := the frame in which this vector exists.
      name := The human readable name of this point
      val := if exists, will take the x, y, and z terms of whatever is put here
             and use those as the seed for this vector. 
      x := the x value
      y := the y value
      z := the z value
      fixed := if true, initializes indices as parameters/constants,
               otherwise variables/initial conditions.
      make_unit := if true, will make a vector from the given parameters, and
                   return the normalized version thereof.
                   otherwise, will enforce a constraint on vector. 
    """

    if make_unit:
        return model.insert(Norm(vector(name=name + ".vector",
                                        val=val,
                                        x=x,
                                        y=y,
                                        z=z,
                                        fixed=fixed)),
                            name=name)
    else: 
        x, y, z = make_xyz(name, val, x, y, z, fixed)
        w = const(0.0) 

        if isinstance(val, ID):
            assert_frames_equal(name, val.frame, frame)

        vec = model.insert(Unit_Vector(x=x,y=y,z=z,w=w,parent_frame=frame), name)
        constrain(length(vec).equals(1.0),
                  name=f'unit-vec-len({name})')

        return vec

@public
@within_model_context
def rot_matrix(model : Model, name : str,
               val : Union[None, MVal] = None, 
               x_axis : Union[None, Dict, MVal] = None, 
               y_axis : Union[None, Dict, MVal] = None,
               z_axis : Union[None, Dict, MVal] = None,
               make_unit : bool = False,
               fixed : bool = False
               ) -> MVal[SOMat3x3]:
    """
    creates some rotation matrix 

    params:
      model := our parent model
      name := The human readable name of this point
      x_axis := The unit_vector for the new x-axis 
      y_axis := The unit_vector for the new y-axis 
      z_axis := The unit_vector for the new z-axis
      make_unit := if true, will make the input into a unit vector, otherwise
                   will enforce that the input is a unit vector.
      fixed := if true generates as parameters, otherwise generates as variables.
    """

    def make_axis(suffix, val):

        nonlocal name
        nonlocal fixed
        nonlocal make_unit

        if isinstance(val, ID):
            val = id_to_mval(val)
            val.add_name(f'{name}{suffix}', is_rel=True)
            return val
        elif isinstance(val, Dict):
            if not ('make_unit' in val):
                val['make_unit'] = make_unit
            
            if not ('fixed' in val):
                val['fixed'] = fixed

            return unit_vector(name + suffix, **val)

        elif val == None:
            return unit_vector(name + suffix, make_unit=make_unit, fixed=fixed)
        else:
            raise Exception("Unsupported type for rot_matrix axis.")

    if val != None:
        x_axis = val.c1
        y_axis = val.c2
        z_axis = val.c3

        for axis, c in [(x_axis,'x'), (y_axis,'y'), (z_axis,'z')]:
            if isinstance(axis, ID):
                axis = id_to_mval(axis)
                axis.add_name(f'{name}.{c}_axis', is_rel=True) 
    else:
        x_axis = make_axis(".x_axis",x_axis)
        y_axis = make_axis(".y_axis",y_axis)
        z_axis = make_axis(".z_axis",z_axis)

    constrain(length(x_axis).equals(1.0),
              name=f'rot-axis-len({name}.x_axis)')
    constrain(length(y_axis).equals(1.0),
              name=f'rot-axis-len({name}.y_axis)')
    constrain(length(z_axis).equals(1.0),
              name=f'rot-axis-len({name}.z_axis)')

    constrain(x_axis.dot(y_axis).equals(0.0),
              name=f'rot-dot({name},x,y)')
    constrain(y_axis.dot(z_axis).equals(0.0),
              name=f'rot-dot({name},y,z)')
    constrain(z_axis.dot(x_axis).equals(0.0),
              name=f'rot-dot({name},z,x)')

    constrain(x_axis.cross(y_axis).equals(z_axis),
              name=f'rot-cross({name})')

    return model.insert(SOMat3x3(x_axis, y_axis, z_axis), name = name) 
    
   

@public
@within_model_context
def axial_rot_matrix(model : Model, name : str, 
                     axis : Union[None, Dict, MVal] = None,
                     angle : Union[None, float, MVal] = None, 
                     fixed : bool = False,
                     make_unit : bool = False,
                     initial_condition : Union[None, Dict, MVal] = False) -> MVal[SOMat3x3]:

    """
    creates a rotation given an axis of rotation and an angle to shift. 

    params:
     model := our parent model
     name := The human readable name of this point
     axis := the vector along which we rotate
     angle := The angle by which we rotate
     fixed := if true generates as parameters, otherwise generates as variables.
    """

    if axis == None:
        axis = unit_vector(name + ".axis",fixed=fixed, make_unit=make_unit)
    elif isinstance(axis, ID):
        axis = id_to_mval(axis)
        axis = norm(axis)
        axis.add_name(name + ".axis", is_rel = True) 
    elif isinstance(axis, dict): 
        if not ('make_unit' in axis):
           axis['make_unit'] = make_unit
           
        if not ('fixed' in axis):
            axis['fixed'] = fixed

        axis = unit_vector(name + ".axis",**axis)
    else:
        raise Exception("Unsupported axis type")

    make_var = fixed_var if fixed else mobile_var

    angle = make_var(name + ".angle", angle)

    # The short form method of generating the relevant axial rotation matrix
    # most of this should probably be simplified out ... rather a lot.

    val = (identity(3) + (crossmat(axis) * sin(angle))
           + ((crossmat(axis) @ crossmat(axis)) * (const(1.0) - cos(angle))))

    """
    # Shorthand used here pulled from pg 30, eq 2.16 of the kinematics text
    # book. We should be able to reduce the above to this
    
    s = sin(angle)
    c = cos(angle) 
    v = const(1) - cos(angle)
    
    w1 = axis.x
    w2 = axis.y
    w3 = axis.z 
    
    # Yeah, this is a bit much,
    return rot_matrix(name, 
                  x_axis = dict(x= (w1.sqr() * v) + c,
                                y= (w1 * w2 * v) + (w3 * s),
                                z= (w1 * w3 * v) - (w2 * s)),
                  y_axis = dict(x= (w1 * w2 * v) - (w3 * s),
                                y= (w2.sqr() * v) + c,
                                z= (w2 * w3 * v) + (w1 * s)),
                  z_axis = dict(x= (w1 * w3 * v) + (w2 * s),
                                y= (w2 * w3 * v) - (w1 * s),
                                z= (w3.sqr() * v) + c),
                  fixed = fixed,
                  make_unit = make+unit)
    """

    return rot_matrix(name,val=val, fixed=fixed, make_unit=make_unit) 


@public
@within_model_context
def rigid_matrix(model : Model, name : str,
                 rot_mat : Optional[MVal] = None,
                 pos_offset : Optional[MVal] = None,
                 fixed : bool = False) -> MVal[SEMat4x4]:
    """
    creates a rigid transform matrix given some offset and position.

    params:
      model := our parent model
      name := The human readable name of this point
      rot_matrix := The rotation matrix in this rigid transform
      pos_offset := The position offset in this transform. 
      fixed := if true generates as parameters, otherwise generates as variables.
    """

    if rot_mat == None:
        rot_mat = rot_matrix(name + ".rot_mat", fixed = fixed)
    else:
        rot_mat.add_name(name + ".rot_mat", is_rel=True)

    if pos_offset == None:
        pos_offset = point(name + ".pos_offset", fixed = fixed)
    else:
        pos_offset.add_name(name + ".pos_offset", is_rel=True)

    return model.insert(SEMat4x4(c1= rot_mat.c1,
                                 c2= rot_mat.c2,
                                 c3= rot_mat.c3,
                                 c4= pos_offset),
                        name = name)
        
@public
@within_model_frame_context
def line(model : Model, frame : Frame, name : str,
         point : Optional[MVal] = None,
         direction : Optional[MVal] = None,
         other_point : Optional[MVal] = None,
         moment : Optional[MVal] = None, 
         fixed : bool = False) -> MVal[Line]:
    """
    a line with a direction, rotations around lines are specified with the
    right hand rule. 
    
    params:
      model := our parent model
      frame := the frame this line is in
      name := The human readable name of this point
      point := a point on the line
      direction := the direction the line is pointed in, as a unit vector
      other_point := a secod point on the line.
      moment := the moment vector of the line. 
      fixed := if true generates as parameters, otherwise generates as variables.
    """
    
    # the options for defining a line

    if direction == None:
        if (point != None) and (other_point != None):
           direction = norm(other_point - point)
           direction.add_name(name + ".dir", is_rel=True) 
        else:
           direction = unit_vector(name + ".dir", fixed=fixed)

    if moment == None:
        if (point != None):
           moment = point.cross(direction)
           moment.add_name(name + ".moment", is_rel=True) 
        else:
           moment = vector(name + ".moment", fixed=fixed)

    constrain(moment.dot(direction).equals(0.0),
              name=f'dir-mom-ortho({name})')

    assert_frames_equal(name,direction.frame,frame)
    assert_frames_equal(name,direction.frame,frame)

    return  model.insert(Line(parent_frame=frame,
                            direction=direction,
                            moment=moment),
                       name=name)

@public
@within_model_frame_context
def point_on_line(model : Model, frame : Frame, line : MVal) -> MVal[Point]:
    """
    Just a helper to get an arbitrary point on a line. Will be the point
    on the line closest to the origin. 
    """

    return point(name = line.name + ".point",
                 val = (cross(line.direction, line.moment)
                        / dot(line.direction, line.direction)))



@public
@within_model_context
def screw_matrix(model : Model, name : str,
                 axis : Optional[MVal] = None,
                 angle : Union[None, float, MVal] = None,
                 offset : Union[None, float, MVal] = None, 
                 fixed : bool = False) -> MVal[SEMat4x4]:
    """
    creates a rigid transform matrix given some offset and position.

    params:
      model := our parent model
      name := The human readable name of this point
      axis := The line along which we rotate
      fixed := if true generates as parameters, otherwise generates as variables.
    """

    if axis == None:
        axis = line(name + ".axis", fixed=fixed)
    else:
        axis.add_name(name + ".axis", is_rel = True) 

    make_var = fixed_var if fixed else mobile_var

    angle = make_var(name + ".angle", angle)
    offset = make_var(name + ".offset", offset)

    rot_mat = axial_rot_matrix(name + ".rot_mat",
                               # The axis in this case is the direction
                               # vector
                               axis=axis.direction,
                               angle=angle,
                               fixed=fixed)

    point_transform = extendmat(identity(3) - rot_mat) @ point_on_line(axis)

    offset = point_transform + axis.direction * offset

    return rigid_matrix(name, rot_mat=rot_mat, pos_offset=offset, fixed=fixed)  


@public
@within_model_context
def ref_frame(model : Model, name : str,
          rigid_mat : Optional[MVal] = None
         ) -> MVal[Frame]:
    """
    creates a rigid reference frame

    params:
      model := our parent model
      name := The human readable name of this point
      rigid_matrix := The full rigid rtansformation matrix, if this exists the
                      rotation and position matrices are ignored. 
      fixed := if true generates as parameters, otherwise generates as variables.
    """

    if rigid_mat == None:
        rigid_mat = rigid_matrix(name + ".matrix")
    else:
        rigid_mat.add_name(name + ".matrix", is_rel=True) 

    # This constraint here is what makes our tool much easier to work with,
    # the assumption that at the initial conditions every single reference
    # frame is identical. 
    constrain(rigid_mat.at_initial.equals(identity(4)),
              name = f'frame-init({name})')

    return model.insert(Frame(c1=rigid_mat.c1,
                             c2=rigid_mat.c2,
                             c3=rigid_mat.c3,
                             c4=rigid_mat.c4),
                        name=name)

@public
@within_model_context
def default_frame(model : Model) -> MVal[Frame]:
    """
    get the default reference frame for the given model. 
    """
    return model.default_frame

@public
@within_model_context
def rigid_body(model : Model, name : str,
               frame : Optional[MVal] = None, 
               rigid_mat : Optional[MVal] = None,
               exists : Union[None, bool, MVal] = None) -> MVal[Frame]:
    """
    creates a rigid transform matrix given some offset and position.

    params:
      model := our parent model
      name := The human readable name of this point
      frame := The rigid frame this body is part of, overrides next 3 inputs if
               it exists. 
      rigid_matrix := The full rigid rtansformation matrix, if this exists the
                      rotation and position matrices are ignored.
      exists := If true, this is a real thing, if false, it isn't, none means
                maybe
    """

    if exists == None:
        exists = param(name + ".exists", bool)
    elif exists == True:
        exists = const(True)
    elif exists == False:
        exists = const(False)
    else:
        exists.add_name(name + ".exists", bool) 

    if frame == None:
        if rigid_mat != None:
           frame = ref_frame(name + ".frame", rigid_mat)
        else:
           frame = ref_frame(name + ".frame")
    else:
        frame.add_name(name + ".frame", is_rel=True) 

    return model.insert(Body(exists=exists,
                             c1=frame.c1,
                             c2=frame.c2,
                             c3=frame.c3,
                             c4=frame.c4),
                        name=name) 
                             
@public
@within_model_context
def frame_transform(model : Model, name : str, 
                    from_frame : MVal[Frame],
                    to_frame : MVal[Frame]) -> MVal[FrameTransform]:
    """
    retrieves a frame transform matrix given two frames, basically just compose
    and multiply matrices. 

    params:
      model := our parent model
      name := The human readable name of this point
      from_frame := frame we are transforming from
      to_frame := frame we transform to 
    """

    trans = from_frame @ to_frame.inverse

    constrain(trans.at_initial.equals(identity(4)),
              name=f'frmtrans-init({name})')

    return model.insert(FrameTransform(c1 = trans.c1,
                                       c2 = trans.c2, 
                                       c3 = trans.c3, 
                                       c4 = trans.c4,
                                       parent_frame = from_frame,
                                       target_frame = to_frame),
                        name=name) 
    

          
@public
@within_model_context
def anchor(model : Model, name : str,
           body : Body,
           location : Union[None, Dict, MVal] = None,
           x : Union[None, float, MVal] = None,
           y : Union[None, float, MVal] = None,
           z : Union[None, float, MVal] = None,
           direction : Optional[MVal] = None,
           exists : Union[None, bool, MVal] = True) -> MVal[Anchor]:
    """
    creates an anchor on a particular body.

    params:
      model := our parent model
      name := The human readable name of this point
      body := The rigid body this object attaches to.
      location := either a dict{x,y,z} or an MVal with the location of the
                  anchor in the frame.
      direction := if non-zero, the direction in which this anchor is pointing.
                   if none is specified, it is assumed this should be the
                   zero vector. 
    """
    
    if exists == None:
        exists = param(name + ".exists", bool)
    elif exists == True:
        exists = const(True)
    elif exists == False:
        exists = const(False)
    else:
        exists.add_name(name + ".exists", is_rel=True) 

    constrain(exists.implies(body.exists),
              name=f'obj-on-body({name},{body.name})')

    if isinstance(location, ID):
        location = id_to_mval(location)
        location.add_name(name + ".loc", is_rel=True)
    elif (location == None) or isinstance(location, dict):
        if location == None:
           location = dict()

        if not ('x' in location):
           location['x'] = x

        if not ('y' in location):
           location['y'] = y

        if not ('z' in location):
           location['z'] = z

        location['frame'] = body 
        location['name'] = name + ".loc"
        location['fixed'] = True

        location = point(**location)
    else:
        raise Exception("Invalid location type")

    if direction == None:
        direction = vector(frame=body,
                           name=name + ".dir",
                           x=0.0, y=0.0, z=0.0,
                           fixed=True)
    else:
        direction.add_name(name + ".dir", is_rel=True) 

    handle = vector(frame=body,
                    name = name + ".handle",
                    fixed=True)

    # the handle should be some arbitrary vector that's perpendicular to
    # direction. That way we can get the relative rotation of the vector
    # with respect to its initial condition later on. 
    constrain(length(handle).equals(length(direction)),
              name='dir-hand-len({name})')
    constrain((dot(direction, handle)).equals(0.0),
              name=f'dir-hand-ortho({name})')

    return model.insert(Anchor(x=location.x,
                               y=location.y,
                               z=location.z,
                               w=location.w,
                               parent_frame=body,
                               exists=exists,
                               direction=direction,
                               handle=handle),
                        name=name)
    

@public
@within_model_context
def hinge(model : Model, name : str,
          from_frame : MVal[Frame],
          to_frame : MVal[Frame],
          axis : Optional[MVal] = None, 
          angle : Optional[MVal] = None,
          min_angle : Optional[MVal] = None,
          max_angle : Optional[MVal] = None,
          exists = True) -> MVal[Hinge]:
    """
    defines a hinge betrween 

    params:
      model := our parent model
      name := The human readable name of this point
      from_frame := frame we are transforming from
      to_frame := frame we transform to 
      axis := fixed with respect to parent frame
      angle := the angle of the hinge at any given point. 
    """

    offset = const(0.0)
    # TODO :: Add boundaries on the angles acceptable
    if min_angle == None:
        min_angle = parameter(name + ".min_angle", float)
    else:
        min_angle.add_name(name + ".min_angle", is_rel=True)

    if max_angle == None:
        max_angle = parameter(name + ".max_angle", float)
    else:
        max_angle.add_name(name + ".max_angle", is_rel=True)

    if angle == None:
        angle = var(name + ".angle", 0.0, max_angle, min_angle)
    else:
        angle.add_name(name + '.angle', is_rel=True)

    if axis == None:
        axis = line(name + ".axis",frame=from_frame, fixed=True)
    else:
        axis.add_name(name + ".axis", is_rel=True)

    constrain(angle.is_between(min_angle, max_angle),
              name=f'bounded-angle({name})')
    constrain(angle.at_initial.equals(0.0),
              name=f'init-angle({name})') 

    ft = frame_transform(name + ".transform", from_frame, to_frame)
    scmat = screw_matrix(name + ".screw", axis, angle, 0.0)

    # We want to ensure that the frame transform and the screw matrix we
    # choose here are properly unified. 
    constrain(scmat.c1.equals(ft.c1),
              name=f'hinge-ftrans-eq({name},c1)')
    constrain(scmat.c2.equals(ft.c2),
              name=f'hinge-ftrans-eq({name},c2)')
    constrain(scmat.c3.equals(ft.c3),
              name=f'hinge-ftrans-eq({name},c3)')
    constrain(scmat.c4.equals(ft.c4),
              name=f'hinge-ftrans-eq({name},c4)')

    return model.insert(Hinge(c1 = scmat.c1,
                              c2 = scmat.c2, 
                              c3 = scmat.c3, 
                              c4 = scmat.c4, 
                              parent_frame = from_frame,
                              target_frame = to_frame,
                              axis=axis,
                              angle=angle,
                              exists=exists,
                              min_angle=min_angle,
                              max_angle=max_angle),
                        name=name)

    
   

@public
@within_model_context
def slide(model : Model, name : str, 
          from_frame : MVal[Frame],
          to_frame : MVal[Frame],
          axis : Optional[MVal] = None, 
          offset : Optional[MVal] = None,
          fixed : bool = False) -> MVal[Hinge]:
   """

   params:
     model := our parent model
     name := The human readable name of this point
     from_frame := frame we are transforming from
     to_frame := frame we transform to 
     axis := the line around which we slide
     offset := the shift between the two points. 
     fixed := if true generates as parameters, otherwise generates as variables.
   """
   raise NotImplementedError

# get_frame_transform (from, to)
#     useful for catching

# add frame (given new axis and offset, default is that you have a variable
#     rotation mat lwrix and point, but are constrained so that its a valid
#     rotmatrix, and point, and their,
#
#     rot_matrix = constrain a == b 

# add body ( ... ) ... 

# Will need some other functions, like, "contains" and stuff.
