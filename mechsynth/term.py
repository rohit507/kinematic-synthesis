from typing import *
from abc import *
from mechsynth.util import * # type: ignore
from dataclasses import dataclass, replace, asdict # type: ignore
from public import public

"""
This module defines the notion of a Term, a type of fixed container that we
use to abstract over notions of expressions and symbols.

Basically, you can define a class for each type of element in an abstract
syntax tree, and then when we later create the notion of an identifier context
`tmap` lets us smoothly convert between representations of expressions that are
split up at a fine grain.

Take a look at `mechsynth/test/test_term.py` and
`mechsynth/test/test_context.py` for a few simple examples of how to make and
use terms. 
"""

_A = TypeVar('_A')
_B = TypeVar('_B')
_C = TypeVar('_C')


TermType = TypeVar('TermType') # type:ignore 
"""
The type variable we export and one should use for the polymorphic elements
within a term.
"""
public("TermType")

@public
@dataclass(frozen=True)
class Term(Generic[TermType], ABC):
  """
  Class for types which can be mapped to contain a different type of element.

  Because of shenanigans with state and mutability, this is probably closer
  to Haskell's Traversable than it is to Haskell's Term. 
  """

  @abstractmethod
  def __tmap__(self, f):
      # type: ignore
      """
      Create a new object that converts its internally stored elements to
      the appropriate type.

      Note :: For some reason mypy can't properly infer the type of this despite
              me spelling it out. I don't know why, just use the type
              `(Term[A], Callable[[A],B]) -> Term[B]`
      """
      pass

  @abstractmethod
  def __tzip__(self, others):
      # type: ignore
      """
      zips the various elements together and produces a new terms with
      relevant tuples.
      """
      pass
  
  @property
  @abstractmethod
  def __attrs__(self):
      """
      dict from field names to their types
      """

@public
def term(cls):
  """
  A decorator that will make your class a term, with some given type variable
  acting as the flag, which tells us whether __tmap__ should modify it.

  Used as follows:

  ::
    from mechsynth.term import *

    @term
    class Foo() :
        fst : TermType
        snd : TermType

  The decorator will fill in (a kinda stupid) generic __tmap__. If you want
  something more complex, like having the terms compose like haskell's
  functors, then you're probably better off defining things manually.

  See `test\test_context.py` for futher examples of using the
  decorator and `mechsynth\symbolic\value.py` for some examples of doing it
  manually for the compositional case. 

  This will only make __tmap__ iterate into those elements which have the type
  `TermType`.

  TODO :: Figure out how to make type annotations work with this. They're
         pretty non-existent right now.  
  """

  setattr(cls,'__hints',dict())


  # NOTE :: The fuck, I can use the decorator to alias a function as a
  #        property and that will propagate when I pass shit around !?
  #        what sort of insane ...
  
  @property
  def __attrs__(self): 
    nonlocal cls
    hints = None
    if not self.__class__.__name__ in getattr(cls,'__hints'): 
      hints = dict()
      for t in self.__class__.mro():
        hints.update(get_type_hints(t))
      getattr(cls,'__hints')[self.__class__.__name__] = hints
    else:
      hints = getattr(cls,'__hints')[self.__class__.__name__]

    return hints

  def __tmap__(self, f):

    kargs = dict()               # Key-args we assemble

    hints = self.__attrs__

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

    hints = self.__attrs__

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

  if not hasattr(cls, "__tmap__"):
    setattr(cls, "__tmap__", __tmap__)
  if not hasattr(cls, "__tzip__"):
    setattr(cls, "__tzip__", __tzip__)
  if not hasattr(cls, "__attrs__"):
    setattr(cls, "__attrs__", __attrs__)
  Term.register(cls)
  return dataclass(cls, frozen=True)

@public
def map_accum(f    : Callable[[_A, _B], Tuple[_B, _C]],
              init : _B,
              v    : Term[_A]) -> Tuple[_B, Term[_C]] :
  """
  Allow you to map over a functor while accumulating a value.

  Parameters:
    f    :: (a,b) -> (b, c)
    f    := The function that accumulates a value b, and produces an output c
    init := The initial value for b
    v    := The functor we're mapping over. 
  """

  accum : _B = init

  def helper(a : _A) -> _C:
    nonlocal accum
    accum, c = f(a, accum)
    return c

  out = v.__tmap__(helper)
  return (accum, out)

_D = TypeVar('_D')
_E = TypeVar('_E')

@public
def tfold(f : Callable[[_D, _E], _E], init : _E, v : Term[_D]) -> _E:
  """
  Accumulate each term of the functor somehow, produce the result.

  Parameters:
    f    :: (a,b) -> b
    f    := Accumulator function
    init := Starting value of accumulator
    v    := Term we're accumulating over
  """

  def helper(a : _D, b: _E) -> Tuple [_E,_C] :
    return (f(a,b), None)

  acc, out = map_accum(helper, init, v)

  return acc 

@public
def tlen(a : Term[_A]) -> int:
    """
    Counts the number of elements in a functor.
    """
    def cnt(a : Any, b : int) -> int : return (b + 1)

    return tfold(cnt, 0 ,a)

@public
def tlist(a : Term[_A]) -> List[_A]:
    """
    Makes a list of all the elements in the functor.
    """

    def cns(a : _A, b : List[_A]) -> List[_A] :
      b.append(a)
      return b

    return tfold(cns,list(),a) 

@public
def tzip(*args) -> Term[Tuple[_E]]:
  return args[0].__tzip__(args[1:])

@public
def tzipwith(f, *args) -> Term[_A]:
  return tmap(lambda a: f(*a), tzip(*args))

@public
def tmap(f : Callable[[_D], _E], a : Term[_D]) -> Term[_E]:
    """
    Takes an immutable object of some type and returns a new immutable object
    with transformed sub elements of that type.

    Parameters:
      f := The function we'll use to map over our object.
      a := The object we're going to map over. 
    """

    # A cast because the type checker can't actually work with the
    # function
    return a.__tmap__(cast(Callable[[_E],_E], f))
