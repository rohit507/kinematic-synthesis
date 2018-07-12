from typing import *
from mechsynth.context           import * # type: ignore
from mechsynth.symbolic.value    import * # type: ignore
from mechsynth.symbolic.geom     import * # type: ignore 
from mechsynth.symbolic.object   import * # type: ignore
from mechsynth.symbolic.operator import * # type: ignore
from mechsynth.symbolic.relation import * # type: ignore
from mechsynth.errors            import * # type: ignore
from dataclasses import *
from uuid import uuid4, UUID
import itertools
import inspect
import wrapt
import pprint
from contextlib import contextmanager
import contextvars as cv
from infix import shift_infix as infix
from public import public

from pprint import pformat
"""
This module defines a symbolic model of a kinematic system. It allows the
(hopefully) easy definition and description of systems at various levels of
ambiguity, as well as the ability to construct algebras over this symbolic
language. 
"""

# TODO :: get all the following using inspect or something

__value_terms__ = [Value, Param, Constant, Control, Variable, 
                     Mat3x1, Mat4x1, Mat3x3, Mat4x4]
"""
A list of terms from the 'mechsynth.symbolremove existential pquantifiericp.value' module
"""

__geom_terms__ = [HasFrame, Point, Vector, Unit_Vector, SOMat3x3, SEMat4x4,
                    Frame, FrameTransform, InFrame, Line]
"""
A list of terms from the 'mechsynth.symbolic.value' module
"""

__object_terms__ = [Body, HasBody, Anchor, Hinge]
"""
A list of classes from the 'mechsynth.symbolic.object' module
"""

__operator_terms__ = [Add, Sub, Negate, Mul, Div, Invert, Sqrt, Pow, Sin, Cos,
                      Tan, Asin, Acos, Atan, Atan2, Dot, Cross , And,
                      Or, Xor, Implies, Iff, Not, IfThenElse, Eq, Neq, LessThan,
                      GreaterThan, LessThanEq, GreaterThanEq, Norm, MatMul,
                      Transpose, Mag, Determinant, CrossMat,
                      Between, Within, Dist]
"""
A list of classes from the 'mechsynth.symbolic.operator' module
"""

__relation_terms__ = [GetMember, AtInitial]
"""
A lot of classes from the 'mechsynth.symbolic.operator' module
"""

__language_terms__ = frozenset(itertools.chain(__value_terms__,
                                               __geom_terms__,
                                               __object_terms__,
                                               __operator_terms__,
                                               __relation_terms__))
"""
The full language of terms we want our model to be able to support.
"""

@public
def model_algebra(cls):
    """
    A decorator to build an algebra over our model. Nothing super serious.
    """
    return term_algebra(__language_terms__)(cls)

_VT = TypeVar('_VT',bound=Any)

model = cv.ContextVar('model')
frame = cv.ContextVar('frame')
model_context_flag = cv.ContextVar('model_context_flag')

@public
@wrapt.decorator
def within_model_context(wrapped, instance, args, kwargs):
    """
    Decorator for model only functions.

    Checks whether we're in a model context. If we're not, it throws an error.
    If we are, adds some context params to kwargs.
    """

    vm = None
    ctxt = cv.copy_context()

    if (('model' in kwargs) and ('frame' in kwargs)):
        pass
    elif ctxt.get(model_context_flag) :
        vm = ctxt[model]
    else:
        raise NotInModelBuildContextError()

    return wrapped(vm, *args, **kwargs)

@public
@wrapt.decorator
def within_model_frame_context(wrapped, instance, args, kwargs):
    """
    Decorator for model only functions.

    Checks whether we're in a model context. If we're not, it throws an error.
    If we are, adds some context params to kwargs.
    """

    vm = None
    vf = None
    ctxt = cv.copy_context()

    if (('model' in kwargs) and ('frame' in kwargs)):
        pass
    elif ctxt.get(model_context_flag) :
        vm = ctxt[model]
        if ('frame' in kwargs):
            vf = kwargs['frame']
            del kwargs['frame'] 
        else:
            vf = ctxt[frame]
    else:
        raise NotInModelBuildContextError()
    return wrapped(vm, vf, *args, **kwargs)

@public
class MVal(ID[_VT]):
    """
    This is just a wrapper for IDs from a model, it only exists so I have a
    class to hang the various python operators and wrappers from.
    """

    def __init__(self, *args, **kargs):
        self.next_flag = False
        super().__init__(*args, **kargs)

    def __hash__(self): return super().__hash__()


    def is_equal_to(self,other):
        """
        So, MVal's use the rich "eq" operation operation to allow for easier
        construction of equations.

        This is_equal_to operation is just how we access the same functionality.
        """
        return super().__eq__(other)

    def set_next(self, val, color="blue:white:blue"):
        if 'next' in self:
            raise Exception("term already has next element, cannot set another.")
        else:
            n = None

            #Make sure we insert expressions and pass IDs through as needed 
            if type(val) == MVal:
                n = val
            elif isinstance(val,Term):
                if self.is_relevant:
                    n = self.parent_context.insert(val, name=self.name)
                else:
                    n = self.parent_context.insert(val)
            else:
                raise TypeError("cannot set value of this type")

            for name in self.other_names:
                n.add_name(name)

            if type(self.val) == Constant and (not self != n):  
                raise Exception("term already has next element, cannot set another.")

            if type(n) != MVal:
                raise Exception("invalid next element")
        
            if n != self:
                n.add_provenance(self)
                self.next_flag = True
                self['next'] = n
                self.next_flag = False
                self['next_col'] = color
                return True
            else:
                return False

            self.set_eq(n)

    def __setitem__(self, k, v) -> None:
        """
        Set a value in the Identifier's internal key value store.

        It is recommended that there exists a schema for the type of value
        associated with any given key. We store them without explicit type
        information and they should be treated as such. 
        """
        if k == 'flag' and not self.next_flag:
            raise Exception('Trying to set \'next\' outside of set_next')

        self.parent_context._id_map[self.uuid].key_vals[k] = v

    @property
    def has_next(self):
        return ('next' in self)

    @property
    def final(self):
        if 'next' in self:
            return id_to_mval(self['next']).final
        else:
            return self

    def set_eq(self, other):
        """
        Create an equality set, and make sure that all members share it.

        This becomes important when we work with variables, and we will
        manipulate constraints to add them to the equality set 
        """

        assert type(other) == MVal

        ss = None
        so = None

        if 'eq_set' in self:
            ss = self['eq_set']
        else:
            ss = set([self])

        if not other in ss: 
            if 'eq_set' in other:
                so = other['eq_set']
            else:
                so = set([other])

                ss.update(so)
                for t in ss: 
                    t['eq_set'] = ss


    def __add__(self,other):
        return self.parent_context.insert(Add(self,other))

    def add(self,other):
        return self.__add__(other)

    def __sub__(self,other):
        return self.parent_context.insert(Sub(self,other))

    def sub(self,other):
        return self.__sub__(other)

    def __mul__(self,other):
        return self.parent_context.insert(Mul(self,other))

    def mul(self,other):
        return self.__mul__(other)

    # TODO ::  //	object.__floordiv__(self, other)

    def __truediv__(self,other):
        return self.parent_context.insert(Div(self,other))

    def __mod__(self,other):
        return self.parent_context.insert(Mod(self,other))

    def __pow__(self,other):
        return self.parent_context.insert(Pow(self,other))

    def pow(self,other):
        return self.__pow__(other)

    def sqr(self):
        return self.__pow__(2.0)

    def sqrt(self):
        return self.__pow__(0.5)

    def __and__(self,other):
        return self.parent_context.insert(And(self,other))

    def __xor__(self,other):
        return self.parent_context.insert(Xor(self,other))
    
    def __or__(self,other):
        return self.parent_context.insert(Or(self,other))

    def __neg__(self):
        return self.parent_context.insert(Negate(self))

    @property
    def negation(self):
        return self.__neg__()

    def implies(self,other): 
        return self.parent_context.insert(Implies(self,other))

    def __abs__(self):
        return self.abs()

    def __pos__(self):
        """
        Marking something as positive doesn't actually change it. 
        """
        return self

    def abs(self,other):
        return self.parent_context.insert(Mag(self))

    def __invert__(self):
        return self.parent_context.insert(Invert(self))

    @property
    def inverse(self):
        return self.__invert__()

    def __matmul__(self,other):
        return self.parent_context.insert(MatMul(self,other))

    # TODO :: Sigh ... this breaks so much shit v_v i'd love to find a way to
    #        between this and __bool__ find a way to make this work ...
    #def __eq__(self,other):

    def equals(self,other):
        return self.parent_context.insert(Eq(self,other))

    def not_equals(self,other):
        return self.parent_context.insert(Neq(self,other))

    def __lt__(self,other):
        return self.parent_context.insert(LessThan(self,other))

    def __gt__(self,other):
        return self.parent_context.insert(GreaterThan(self,other))

    def __le__(self,other):
        return self.parent_context.insert(LessThanEq(self,other))

    def __ge__(self,other):
        return self.parent_context.insert(GreaterThanEq(self,other))

    def dot(self,other):
        return self.parent_context.insert(Dot(self,other))

    def cross(self,other):
        return self.parent_context.insert(Cross(self,other))

    def dist_to(self,other):
        return self.parent_context.insert(Dist(self,other))

    def is_between(self,min,max):
        return self.parent_context.insert(Between(self,min,max))

    def within(self,dist,of):
        return self.parent_context.insert(Within(self,dist,of))
    
    def in_frame(self,target):
        return self.parent_context.insert(InFrame(self,target))
    @property
    def transpose(self):
        return self.parent_context.insert(Transpose(self))

    @property
    def inverse(self):
        return self.parent_context.insert(Invert(self))

    @property
    def at_initial(self):
        return self.parent_context.insert(AtInitial(self))

    @property
    def at_init(self):
        return self.at_initial

    # ## Properties that various terms can have within this language ##

    def __member(self,name):
        """
        If you have a value that has the term already, just get it, otherwise
        get it after some reduction. 
        """
        if hasattr(self.val, name):
            return getattr(self.val, name)
        else:
            return self.parent_context.insert(GetMember(name,self))

    @property
    def x(self):
        return self.__member('x')

    @property
    def y(self):
        return self.__member('y')

    @property
    def z(self):
        return self.__member('z')

    @property
    def w(self):
        return self.__member('w')

    @property
    def c1(self):
        return self.__member('c1')

    @property
    def c2(self):
        return self.__member('c2')

    @property
    def c3(self):
        return self.__member('c3')

    @property
    def c4(self):
        return self.__member('c4')

    @property
    def r1(self):
        return self.__member('r1')

    @property
    def r2(self):
        return self.__member('r2')

    @property
    def r3(self):
        return self.__member('r3')

    @property
    def r4(self):
        return self.__member('r4')

    @property
    def axis(self):
        return self.__member('axis')

    @property
    def rot(self):
        return self.__member('rot')

    @property
    def shift(self):
        return self.__member('shift')

    @property
    def angle(self):
        return self.__member('angle')

    @property
    def offset(self):
        return self.__member('offset')

    @property
    def exists(self):
        return self.__member('exists')

    @property
    def dir(self):
        return self.direction

    @property
    def direction(self):
        return self.__member('direction')

    @property
    def loc(self):
        return self.location

    @property
    def location(self):
        return self.__member('location')

    @property
    def moment(self):
        return self.__member('moment')

    @property
    def frame(self):
        return self.parent_frame

    @property
    def parent_frame(self):
        return self.__member('parent_frame')

    @property
    def target_frame(self):
        return self.__member('target_frame')

    @property
    def point_offset(self):
        return self.__member('point_offset')

    @property
    def rotation_metrix(self):
        return self.__member('rotation_matrix')

@public
def id_to_mval(ident : ID[_VT]) -> MVal[_VT]:
    """
    Takes an identifier and copies it over a new MVal.
    """
    if type(ident) == MVal:
        return ident
    else: 
        return MVal(ident.uuid, ident.parent_context, ident.stored_type, hash(ident))

@public
@dataclass()
class Model(Context):
    """
    This class captures a full model of a rigid body kinematic system with
    various properties and members.

    These properties and members can cover the full range from simple variables
    to rigid bodies with bounding boxes and internal reference frames.

    We assume everything interesting in this model is parameterized by `t`.
    As `t` changes so do the user supplied axes and degrees-of-freedom, and
    we can calculate derivatives with respect to `t`.

    Fields:
      default_frame := The identity frame, the one we keep track of everything
                       else relative to.

      num_frames := the maximum number of frames/bodies we can synthesize/solve
                    over. 

      ref_frames := All the other reference frames / rigid bodies that are
                    to this system.

      objects := Things that exist (or may exist) within the space of the model,
                 like points and hinges.

      parameters := Values that must be assigned some constant value by the
                    solver.

      controls := control parameters, within some bounds, that define the state
                 of the system. 

      variables := possible degrees of freedom that depend on the various
                   control parameters.

      assertions := Constraints that we should be able to verify before runtime,
                    they should be reduced to constants beforehand. 

      assumptions := Things that the model assumes are true at all times. We
                     would like to ensure that the model is valid for the
                     entire space of assumptions.

      constraints := Constraints on the state of the model, the existence of
                     objects for instance.

      guarantees := Things which we wish to be true whenever the assumptions
                    are true and the constraints hold.

      cost_terms := Terms that we ask the solver to minimize over when choosing
                    a design. We probably won't use these directly for a while
                    but having the option is nice at least.

      _local_context := TODO
      _term_language := TODO
   """

    default_frame : MVal[Frame]      = field(repr=False, default=None)
    ref_frames    : Set[MVal[Frame]] = field(repr=False, default_factory=set)
    objects       : Set[MVal]        = field(repr=False, default_factory=set) 
    parameters    : Set[MVal]        = field(repr=False, default_factory=set) 
    controls      : Set[MVal]        = field(repr=False, default_factory=set) 
    variables     : Set[MVal]        = field(repr=False, default_factory=set) 
    assertions    : Set[MVal]        = field(repr=False, default_factory=set) 
    assumptions   : Set[MVal]        = field(repr=False, default_factory=set) 
    constraints   : Set[MVal]        = field(repr=False, default_factory=set) 
    guarantees    : Set[MVal]        = field(repr=False, default_factory=set)
    cost_terms    : Set[MVal]        = field(repr=False, default_factory=set)
    named_vars    : Dict[str,MVal]   = field(repr=False, default_factory=dict)

    # _local_context : cv.Context       = field(repr=False, default_factory=cv.copy_context)

    _term_language : FrozenSet[Type] = field(repr=False,default=__language_terms__)

    @contextmanager
    def build(self):
        """
        A context manager within which you define the system you wish to
        analyze or synthesize.

        TODO :: Make better
        """

        mt = model.set(self)
        ft = frame.set(self.default_frame)
        mcft =  model_context_flag.set(True)

        yield

        model.reset(mt)
        frame.reset(ft)
        model_context_flag.reset(mcft) 

    def __hash__(self):
        return super().__hash__()

    def __post_init__(self):

        ref_frame = self.insert(Frame(
            c1 = Mat4x1(1.0,0.0,0.0,0.0),
            c2 = Mat4x1(0.0,1.0,0.0,0.0),
            c3 = Mat4x1(0.0,0.0,1.0,0.0),
            c4 = Mat4x1(0.0,0.0,0.0,1.0)),
                                name="default_frame")

        self.default_frame = ref_frame
        pass

    def _const(self, val : Union[float,int,bool]) -> MVal[Constant]:
        """
        Create a constant value within this model.
        """
        return self.insert(Constant(val))


    ## TODO :: Optional min-max? 
    def _param(self,
               name : str,
               type_hint : Union[Type, ValType],
               max_bound : Union[None, float] = None,
               min_bound : Union[None, float] = None, 
               provenance : Union[None, MVal, Set[MVal]] = None) -> MVal[Param]:
        """
        Create a new parameter with some given name and type.

        params:
          name := Human readable name
          type_hint := either 'bool', 'float', or 'int'. Used to 
        """

        if name == None or name == 'Param':
            raise Exception('all params must be named')

        e_name = f'{name}_{uuid4().hex[:4]}'

        if type_hint == bool :
            type_hint = ValType.BOOL
        elif type_hint == float :
            type_hint = ValType.REAL
        elif type_hint == int :
            type_hint = ValType.INT

        if not type(type_hint) == ValType:
            raise TypeError("Parameter type hints must be `int`," +
                        " `bool`, or `float`")

        mv = self.insert_fresh(Param(e_name, type_hint), name, provenance)

        if max_bound:
            self.add_constraint(mv <= const(max_bound),
                                name=f'par-max({name})')

        if min_bound:
            self.add_constraint(mv >= const(min_bound),
                                name=f'par-max({name})')

        self.parameters.add(mv)
        return mv

    def _control(self, name : str,
                initial_condition : Union[None, float, MVal] = None,
                max_bound : Union[None, float, MVal] = None,
                min_bound : Union[None, float, MVal] = None, 
                provenance : Union[None, MVal, Set[MVal]] = None) -> MVal[Control]:
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

        if name == None:
            raise Exception('all controls must be named')

        e_name = f'{name}_{uuid4().hex[:4]}'

        if (type(initial_condition) == float):
            initial_condition = self._const(initial_condition)
            initial_condition.add_name(name + '.init', is_rel=True)
        elif(initial_condition == None):
            initial_condition = self._param(name + ".init", float)
        else:
            initial_condition.add_name(name + '.init', is_rel=True)


        mv = self.insert_fresh(Control(e_name, initial_condition),
                                           name, provenance)

        if max_bound:
            self.add_assumption(mv <= max_bound,
                                name=f'cntl-max({name})')

        if min_bound:
            self.add_assumption(mv >= min_bound,
                                name=f'cntl-min({name})')
        
        self.controls.add(mv)
        return mv

    def _variable(self, name : str,
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

        if name == None:
            raise Exception('all variables must be named')

        e_name = f'{name}_{uuid4().hex[:4]}'

        if (type(initial_condition) == float):
            initial_condition = self._const(initial_condition)
            initial_condition.add_name(name + '.init', is_rel=True)
        elif(initial_condition == None):
            initial_condition = self._param(name + ".init", float)
        else:
            initial_condition.add_name(name + '.init', is_rel=True)


        mv = self.insert_fresh(Variable(e_name, initial_condition),
                                           name, provenance)

        if not mv.is_relevant:
            raise Exception('all vars must be relevant')

        if max_bound:
            self.add_assumption(mv <= max_bound,
                                name=f'var-max({name})')

        if min_bound:
            self.add_assumption(mv >= min_bound,
                                name=f'var-min({name})')


        self.variables.add(mv)
        return mv

    def insert_fresh(self,
               val : _VT,
               name : Optional[str] = None,
               provenance : Union[None, MVal, Set[MVal]] = None) -> MVal[_VT]:


        if (type(val) in {Param, Control, Variable}):
            if name == None or name == val.__class__.__name__:
                name = val.name

        if (type(val) in {Param, Control, Variable}):
            if name == None or name == val.__class__.__name__:
                raise NameError

        n_val = tmap(id_to_mval,val)

        mv = id_to_mval(super().insert_fresh(n_val, name, provenance))

        if isinstance(mv.val, Object) and (not mv in self.objects):
            self.objects.add(mv)

        if isinstance(mv.val, Frame) and (not mv in self.ref_frames):
            self.ref_frames.add(mv)

        if (type(val) in {Param, Control, Variable}):
            if val.name in self.named_vars:
                raise NameError
            self.named_vars[val.name] = mv

        return mv
        
    def insert(self,
               val : _VT,
               name : Optional[str] = None,
               provenance : Union[None, MVal, Set[MVal]] = None,
               ) -> MVal[_VT]:

        # This is kuludge to deal with name propagation issues when we
        # decompose types FIXME 
        if (type(val) in {Param, Control, Variable}):
            if val.name in self.named_vars:
                mv = self.named_vars[val.name]
                if name != None and name != 'Param':
                    mv.add_name(name)
                return mv
            else:
                name = val.name.split('_')[0]

        if name == 'Param':
            raise NameError

        def _to_const(a):
          if type(a) == bool:
              return self._const(a)
          elif type(a) == int:
              return self._const(a)
          elif type(a) == float:
              return self._const(a)
          elif a == None:
              raise TypeError
          else:
              return a

        norm_val = tmap(_to_const, val)
        mv = id_to_mval(super().insert(norm_val, name, provenance))

        if isinstance(mv.val, Object) and (not mv in self.objects):
            self.objects.add(mv)

        if isinstance(mv.val, Frame) and (not mv in self.ref_frames):
            self.ref_frames.add(mv)

        return mv

    def __add_w_init(self,
                     field : Set[MVal],
                     val: Union[MVal, _VT],
                     name=None,
                     provenance = None, 
                     init=True):


        if name == None:
            raise NameError

        term = None
        init_name = None if name == None else f'init({name})'
        init_term = None

        if type(val) == MVal or isinstance(val, ID):
            term = id_to_mval(val)
        elif isinstance(val, Term): 
            term = self.insert(val, name, provenance)

        if name != None:
            term.add_name(name)
            term.set_relevant()

        if provenance != None:
            term.add_provenance(provenance) 

        field.add(val)

        if init:
            init_term = self.insert(AtInitial(term), init_name, provenance)
            self.assertions.add(init_term) 

        return None

    def add_assertion(self, val : Union[MVal, _VT], *args, **kwargs):
        """
        Take a statement and add it to the set of model assumptions.

        i.e: Things that are always true about the controls. 
        """
        self.__add_w_init(self.assertions, val, *args, **kwargs)

    def add_assumption(self, val : Union[MVal, _VT], *args, **kwargs):
        """
        Take a statement and add it to the set of model assumptions.

        i.e: Things that are always true about the controls. 
        """
        self.__add_w_init(self.assumptions, val, *args, **kwargs)

    def add_constraint(self, val : Union[MVal, _VT], *args, **kwargs):
        """
        Take a statement and add it to the set of model constraints.

        i.e. Things that are true about the model given some parameters
        """
        self.__add_w_init(self.constraints, val, *args, **kwargs)

    def add_guarantee(self, val : Union[MVal, _VT], *args, **kwargs):
        """
        Take a statement and add it to the set of model requirements.

        i.e. things we wish to choose parameters for such that these are true.
        """
        self.__add_w_init(self.guarantees, val, *args, **kwargs)

    def add_cost(self, val : Union[MVal, _VT], *args, **kwargs):
        """
        Take a statement and add it as a simple term to the cost function. 
        """
        self.__add_w_init(self.cost_terms, val, *args, **kwargs, init=False)


    # TODO :: adding objects/bodies

    # TODO :: We should have some way of just getting the transform between any
    #        two frames.

    # TODO :: 

