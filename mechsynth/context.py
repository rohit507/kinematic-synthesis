from typing import *
from types import *
from abc import *
from uuid import uuid4, UUID
from mechsynth.term import Term, map_accum, tfold, tlen, tmap, tlist, TermType
from mechsynth.errors import *
from mechsynth.util import * 
from inspect import *
from dataclasses import dataclass, field # type: ignore
from toposort import toposort, toposort_flatten # type: ignore
from graphviz import Digraph  # type: ignore
from public import public
from pprint import pformat

"""
This modules defines the notion of ID and Contexts.

Basically, if you have a term (like `Variable(name)` or
`Add(expr_a, expr_b)`) you can use it as a part of a larger expression:

   x = Variable("X")
   foo = Add(expr_a, Mul(expr_b, expr_c))

These expressions can be assertions, guarantees, relationships, or any number
of other relationships we want to describe in an abstract syntax tree.
This module allows you to take expressions written with functors and store
them as a graph that can be analyzed or edited.

A Context gives you a coherent way to handle a large number of terms that are
are all interdependent and form, a great big relationship graph (well, DAG).
When wrapped in something else a context lets you manage all these
definitions and relationships effectively. 

This is all a bit more evident if you look in `mechsynth/test/test_context.py`,
which has a handful of very simple tests and definitions. 
"""

_VT = TypeVar('_VT', bound=Term[Any])

@dataclass
class _Ctxt(ABC):
  """
  An internal class to break a definition cycle in our type signatures.
  """
  name : str
  _ctxt_ident : UUID = field(default_factory=uuid4)
  _val_map : 'Dict[Term[UUID],ID]' = field(default_factory=dict,repr=False)
  _id_map : 'Dict[UUID, _ID_Metadata]' = field(default_factory=dict,repr=False)
  _term_language : FrozenSet[Type] = field(default=frozenset([Term]),
                                          repr=False)


def _validate_lang(lang : FrozenSet[Type]):
  """
  Throws an error if this language is not made up of valid types.

  TODO :: Implement this
  """
  pass

@public
@dataclass(eq=False)
class ID(Generic[_VT]) :
    """
    An identifier which stores a value of a known type. This shouldn't really
    be initialized directly, only created from the Context it will then
    be associated with.

    Type Variables:
      _VT := The Type of the thing this points to, must be a functor. 
    """

    __ident  : UUID
    __parent : _Ctxt
    __type_hint : Type[_VT]
    _hash : int = None


    def __hash__(self):
      """
      Really the only thing that matters for equality here is that we are
      talking about the same identifier in the same context.
      """
      if self._hash == None:
        self._hash = hash(self.__ident.bytes[:4])
      return self._hash# hash((self.__ident,self.__parent))


    def __eq__(self, other) -> bool:
      return (hasattr(other,'_hash')
              and (hash(self) == hash(other)))

    def __repr__(self):

      def flat_str(val):
        """
        Flattens a set of values into a more readable representation. 
        """
        type_name = val.__class__.__name__
        members = list()
        num_flds = len(val.__attrs__)
        
        for fld,typ in val.__attrs__.items():

           fld_nm = fld

           fld_nm = fld_nm[4:] if fld_nm.startswith("exp_") else fld_nm
           fld_nm = "" if len(fld_nm) <= 2 and num_flds <= 2 else fld_nm
           fld_nm = fld_nm[:4] if len(fld_nm) > 4 else fld_nm

           val_str = None

           if typ == TermType:
             val_str = flat_str(getattr(val,fld).val)
           else:
             val_str = repr(getattr(val,fld))

           if fld_nm == "" or num_flds <= 1:
             members.append(val_str)
           else:
             members.append(f'{fld_nm}={val_str}')

        if type_name != 'Constant': 
          return f'{type_name}({", ".join(members)})'
        else:
          return f'{", ".join(members)}'


      members = repr({
        'name' : self.name,
        #'type' : self.__type_hint,
        #'ident' : self.uuid,
        #'relevant' : self.is_relevant,
        'assign' : self.assign_statement,
        #'val' : flat_str(self.val),
        'names' : self.other_names,
      })

      return f'ID({members})'

    def is_equal_to(self,other):
        return self.__eq__(other)

    @property
    def stored_type(self) -> Type[_VT]:
      return self.__type_hint

    @property
    def parent_context(self):
      return self.__parent

    @property
    def name(self) -> str:
      """
      Get the human readable name of this element. If the element was given a
      name when constructed, this will be exactly that name. 
      """
      return self.__parent._id_map[self.__ident].name

    @property
    def is_relevant(self) -> str:
      return self.__parent._id_map[self.__ident].relevant

    def set_relevant(self, val = True):
      self.__parent._id_map[self.__ident].relevant = (
        self.__parent._id_map[self.__ident].relevant or val)

    @property
    def short_name(self) -> str:
      """
      Get the short name of this element. This should be a short random hex
      code that is (probably) unique to this particular object.

      Use only for debug. 
      """
      return self.__parent._id_map[self.__ident].short_name

    @property
    def long_name(self) -> str:
      """
      Get the long name of this element. It should contain the shortnames of
      every element that this value points to.

      Use only for debug
      """
      return self.__parent._id_map[self.__ident].long_name

    @property
    def other_names(self) -> str:

      for cls in self.__parent._term_language:
        self.__parent._id_map[self.__ident].other_names.discard(cls.__name__)

      return self.__parent._id_map[self.__ident].other_names 

      

    def add_name(self,s,is_rel=False):
      self.set_relevant(is_rel)
      self.__parent._id_map[self.__ident].other_names.add(s)

    @property
    def uniq_name(self) -> str:
      """
      Gets a semi-readable (probably) unique name.

      Only use for debug.
      """

      return self.name + "_" + self.short_name

    @property
    def assign_statement(self) -> str:
      """
      An assignment statement that describes this element, in a format that
      is pretty human readable. Basically it's, "<short_name> := <long_name>",
      which is usually enough to let you figure out what's going on.

      Use only for debug. 
      """

      return self.short_name + " := " + self.long_name

    @property
    def val(self) -> 'Term[ID]':
      """
      Get the value this identifier points to. 
      """
      return self.__parent._id_map[self.__ident].value

    @property
    def uuid(self) -> UUID:
      """
      Get the uuid for this particular object.
      """

      return self.__ident

    def add_provenance(self, p : 'ID') -> None:
      """
      Add an element to the set that this one is derived from.

      Note that this does not require the identifier to be from the same
      context as this one, and in fact is the most useful when the identifiers
      are not from the same context. 
      """
      if p == None:
        pass
      elif isinstance(p,ID):  
        self.__parent._id_map[self.__ident].provenance.add(p)
      elif isinstance(p, Set):
        self.__parent._id_map[self.__ident].provenance.update(p)
      else:
        raise TypeError


    def add_provenances(self, p : 'Set[ID]') -> None:
      """
      Add multiple identities to the set that this one is derived from.
      """
      self.add_provenance(p)

    def __getitem__(self, k) -> Any:
      """
      Get a value from the Identifier's key value store, often used to help
      encoding and decoding a system from one identifier class to another.
      """
      return self.__parent._id_map[self.__ident].key_vals[k]

    def __delitem__(self, k) -> Any:
      """
      delete an item from the map.
      """
      return self.__parent._id_map[self.__ident].key_vals.__delitem__(k)

    def __contains__(self, k) -> Any:
      """
      Df o we store a value for some key? 
      """
      return self.__parent._id_map[self.__ident].key_vals.__contains__(k)

    def __setitem__(self, k, v) -> None:
      """
      Set a value in the Identifier's internal key value store.

      It is recommended that there exists a schema for the type of value
      associated with any given key. We store them without explicit type
      information and they should be treated as such. 
      """
      self.__parent._id_map[self.__ident].key_vals[k] = v 


@dataclass
class _ID_Metadata() :
  """
  The set of various pieces of information that are associated with a value in
  some given ID context.

  Parameters:
    name := Human readable name for the variable.
    short_name := A short identifier for this element, usually the first few
                 characters of the UUID.
    long_name := A more detailed name for the element this contains. 
    ident := The identifier that points to this object. 
    value := The value that this metadata object is storing.
    relevant := A relevant value is one that was given an explicit name when
               initialized. Which we are summing means that the user cares
               about it in some fashion, and one should preserve it as
               a context is pruned. 
    other_names := The other names that might be assigned to this object, only
                   the first one will be used for anything, this is mainly for
                   debug and tracking purposes. 
    provenance := A set of identifiers (from any context) that had a part in
                 building this value.
    key_vals := A set of key value pairs associated with this identifier,
               useful for storing data and other things. We'll probably use
               this rather regularly when performing conversions from one
               context to another. 
  """

  name        : str
  short_name  : str
  long_name   : str
  ident       : ID[Term]
  value       : Term
  relevant    : bool 
  other_names : Set[str]      = field(default_factory=set) 
  provenance  : Set[ID]       = field(default_factory=set)
  key_vals    : Dict[str,Any] = field(default_factory=dict) 

@public
@dataclass(eq=False)
class Context(_Ctxt):
    """
    A class for generating typed unique identifiers for use through this
    program. Basically, every Context will produce a stream of UIDs that
    carry type information, some human readable context, and are associated
    with objects of the corresponding type.

    Internal Variables:
      name := Human readable name for this context
      _ctxt_ident := the UUID for this context
      _val_map := A map we can use to find Values that are already in this
                 Context.
      _id_map := A map that allows us to get the metadata associated with a
                 particular identifier.
      _term_language := A set of class objects (all of which must be instances
                       of Term) which values in this context must be members of.
                       Well they must be instances of at least one of the
                       classes. 
    """

    def __eq__(self, other) -> bool:
      return (hasattr(other,'_ctxt_ident')
              and (self._ctxt_ident == other._ctxt_ident))

    # def __del__(self):
    #  print(f'Deallocating context {self.name}')

    def __hash__(self):
      return hash(self._ctxt_ident)

    def in_language(self, val : Term) -> None:
      """
      Checks whether a particular term is within the language of this context.
      Throws an error if the input is incompatible with this context. 

      Parameters:
        val := The term we check for class validity. 
      """
      # Check whether the input is a functor. 
      if not issubclass(type(val), Term) :
        raise NotTermError()

      # Check if we're in the algebra at allows
      if not any(issubclass(type(val),cls) for cls in self._term_language):
        raise TermNotInLanguageError()
      
    def insert_fresh(self,
               val : _VT,
               name : Optional[str] = None,
               provenance : Union[None, ID, Set[ID]] = None) -> ID[_VT]:
      """

      This will insert a value into the Context and provide you a key that
      can be used to retrieve the value. This will **not** check whether the
      value is already part of the context and just give you something fresh.
      For the most part, just don't call this, it's meant for new variables
      and things which might have duplicate names but are actually different
      entities.

      Additionally, insert_fresh does not recurse into the functor given and
      will fail when given a functor with identifiers that are not already
      in the context. 

      NOTE :: Anything added with insert_fresh is *not* going to be part of the
             table we store existing IDs in, and so won't be found when insert
             is called with an indentical element. 

      Parameters:
        val        := The functor we're inserting into this context, containing
                      either further nested functors or identifiers.
        name       := The human-readable name given to this element, will be
                      modified to add the first few chars of the hex UUID.
                      If none is given, will default to the classname + some
                      metadata.
        provenance := The ids which led to the generation of this one. This is
                      just debug metadata.

      Returns:
        The identifier that corresponds to the given value. 
      """

      # Check that all elements of the functor are IDs for elements that can
      # exist within this context.
      self.in_language(val) 

      # Will raise an exception if any element of the input functor is not
      # a valid identifier. If the functor is empty, we're peachy. 
      def verify_ident(a : Any) -> None:
        if isinstance(a, UUID):
          if not (a in self._id_map):
            raise IdentifierDoesNotExistError()
        elif not issubclass(type(a), ID):
          raise NotIdError()
        elif a.parent_context != self :
          raise IncorrectContextError()
        elif not (a.uuid in self._id_map):
          raise IdentifierDoesNotExistError()
        return None

      tmap(verify_ident, val) 

      # Create a new UUID for this ID
      new_uuid   = uuid4()
      type_name  = type(val).__name__
      short_name = new_uuid.hex[:6]

      relevant = False

      # Assemble the potential  identifier with the UUID, parent, and hint
      ident : ID[_VT] = ID(new_uuid, self, type(val)) # type: ignore

      # Generate the name if needed.
      if name == None:
        relevant = False 
        name = type_name
      else:
        relevant = True 

      # create the long name
      long_name = None
      if tlen(val) == 0:
        long_name = repr(val)
      else :
        def collect_short_names(a : ID, b : List[str]) -> List[str]: 
          b.append(a.short_name)
          return b

        params = ",".join(tfold(collect_short_names, list(), val)) #type: ignore
        long_name = type_name + "(" + params + ")"

      # Assemble the metadata
      id_meta = _ID_Metadata(name, short_name, long_name, # type: ignore
                             ident, val, relevant, set([name]) if relevant else set()) 

      # Insert metadata into the table
      self._id_map[new_uuid] = id_meta

      # Add provenance if needed.
      if isinstance(provenance, ID):
        ident.add_provenance(provenance)
      elif isinstance(provenance, Set):
        ident.add_provenances(provenance)
      elif provenance != None:
        raise TypeError("Provenance provided was not of valid type.") 

      # Return the ID we just created
      return ident 

    def insert(self,
               val : _VT,
               name : Optional[str] = None,
               provenance : Union[None, ID, Set[ID]] = None
               ) -> ID[_VT]:
      """
      This will insert a value into the Context and provide you a key that
      can be used to retrieve the value. This will check whether the value
      you're trying to insert already exists, and will just provide the
      existing ID if it does.

      This will also happily recurse if you pass it nested functors, adding
      them all to the context. This means that you can assemble a large
      expression and the insert operation itself will explicitly decompose it
      as necessary. 

      Parameters:
        val        := The functor we're inserting into this context, containing
                      either further nested functors or identifiers.
        name       := The human-readable name given to this element, will be
                      modified to add the first few chars of the hex UUID.
                      If none is given, will default to the classname + some
                      metadata.
        provenance := The ids which led to the generation of this one. This is
                      just debug metadata.

      Returns:
        The identifier that corresponds to the given value. 
      """

      # Check/Insert any sub elements if they can be.
      def validate_elems(a : Any, b : List[ID]) -> Tuple [List [ID], ID]:
        # The normalized identifier. 
        norm_id = None

        # If we get an ID, check that it's already in this context. 
        if isinstance(a, ID):
          if a.parent_context != self :
            raise IncorrectContextError()
          elif not (a.uuid in self._id_map):
            raise IdentifierDoesNotExistError()
          else:
            norm_id = a
        # Otherwise, we can just insert the functor into this context, and
        # rely on the recursive call's various checks. 
        elif (isinstance(a, Term)):
          norm_id = self.insert(a)
        else:
          raise Exception(f'type {type(a)} is not insertable.')

        b.append(norm_id)
        return (b, norm_id)

      # This returns the updated children and the various
      children : List[ID]
      children, norm_val = map_accum(validate_elems, list(), val) # type: ignore
      
      # By usig UIDs instead of objects here, we can handle things
      # like a subtype of ID overloading __eq__
      uid_norm = tmap(lambda x: x.uuid, norm_val)

      out_id = None;

      # If not already in lookup table
      if not (uid_norm in self._val_map):

        # Insert Fresh
        out_id = self.insert_fresh(norm_val, name, provenance) 

        # Add value to the lookup table
        self._val_map[uid_norm] = out_id

      else: 

        # Get the identifier that already exists
        out_id = self._val_map[uid_norm] 

        if name != None:
          out_id.add_name(name, is_rel=True)

        # Add any provenance as needed 
        if isinstance(provenance, ID):
          out_id.add_provenance(provenance)
        elif isinstance(provenance, Set):
          out_id.add_provenances(provenance)
        elif provenance != None:
          raise TypeError("Provenance provided was not of valid type.")

      # Add self to the provenances of children.
      map(lambda x: x.add_provenance(out_id), children) # type: ignore 

      # Return the final output value.
      return out_id # We just flatten the recursive type a bit. 

    @property
    def dependency_map(self) -> Dict[ID, Set[ID]]:
      """
      Make a dictionary of all the identifiers in this context, and their
      dependencies.
      """

      deps : Dict[ID, Set] = dict()

      # Go through and grab all the 
      for meta in self._id_map.values():
        deps[meta.ident] = set(tlist(meta.value))

      return deps

    def topo(self) -> List[Set[ID]]:
      """
      The topological sort of all the terms within this context. Each
      set can be treated as independent of each other, as long as you
      process the sets in list order.
      """

      return toposort(self.dependency_map) 

    def topolist(self) -> List[ID]:
      """
      Gets a list of identifiers in this object in dependency resolution order.
      Specifically, if you pick any element of the list, the identifiers its
      stored value points to would have appreared earlier in the list.

      In this way, you can implement a type-inference, transformation, or
      other analysis pass by just iterating over the list that this function
      outputs in order.

      This function will throw an error if a dependency cycle is detected in
      your context. Though this should not be possible if you only use the
      given interface functions.
      """

      return toposort_flatten(self.dependency_map, sort=False)

    def to_graphviz(self) -> Digraph:
      """
      Generate a diagram of all the elements in the context. Mainly meant for
      debugging, but may also be useful elsewhere.

      TODO :: Make it so you can get graphs of various verbosity as needed?
             Alternately make it so you can just define your own function to
             convert elements to nodes, but frankly that's pretty easy already. 
      """

      dot = Digraph()

      for elem in self.topolist():
        dot.node(elem.short_name, label=elem.long_name)
        for link in tlist(elem.val):
          dot.edge(elem.short_name, link.short_name)

      return dot

    def purge_key(self, keys: Union[str,Set[str]]) -> None:
      """
      Goes through all the identifiers and purges a key or set of keys from
      ever single term's metadata.

      This is mostly for when you want to clean the metadata storage up after
      using various keys for some operation or another. It's not the most
      elegant mechanism, but it'll do.
      """

      if isinstance(keys, str):
        keys = set(keys) 

      for meta in self._id_map.values():
        for key in keys:
          if key in meta.key_vals:
            del meta.key_vals[key]


    def purge_keys(self, keys: Union[str,Set[str]]) -> None:
      self.purge_key(keys)

    def run_algebra(self, alg : '_Term_Algebra') -> Any:
      """
      Evaluate some Term Algebra over this particular context, basically
      run the reduction function in topological order. 
      """

      dirty = True
      alg._init_algebra(self)
      if not hasattr(alg, '_no_run'):
        while dirty:
          alg._init_pass(self)
          for layerset in self.topo():
            for ident in layerset:
              alg._run(ident, ident.val)
          dirty = alg._end_pass(self)  

      return alg._end_algebra(self)


@dataclass
class _Term_Algebra(ABC):
  """
  A term algebra is an object that can be used to iterate over all the
  terms in a context and perform some sort of reduction/analysis/annotation
  over them.

  This class is just an abstract base that defines the minimal necessary
  functions for an algebra.

  Class Vars:
    language := The language of terms this is an algebra over.
    _run_funcs := hacky way to store mapping between types and method attributes.
  """

  language : ClassVar[Dict[Type,str]]


  @abstractmethod
  def _run(self, ident : ID[Term], val : Term[ID]) -> None:
    """
    This function should take an identifier in some context, its value, and
    properly dispatch it to the appropriate reduction functions in reverse
    mro order.

    The `@term_algebra` decorator will define it for you. 

    Parameters:
      ident := The identifier of the term we're reducing.
      val   := The value associated with the above identifier. The system will
               have already run the reduction functions of every term you can
               access recursively from this value. 
    """
    pass

  @abstractmethod
  def _init_algebra(self, context : Context) -> None:
    """
    Initializes the algebra to run as needed.
    """
    pass

  @abstractmethod
  def _init_pass(self, context : Context) -> None:
    """
    runs before each pass of the algebra.
    """
    pass

  @abstractmethod
  def _end_pass(self, context : Context) -> bool:
    """
    Runs after each pass of the algebra.

    It should return a dirty flag. True if another pass needs to be run, false
    otherwise.
    """
    return False 

  @abstractmethod
  def _end_algebra(self, context : Context) -> Any:
    """
    Runs at the end of the algebra, whatever this function returns will be
    returned from the run_algebra function.
    """
    return None


@public
def term_algebra(lang = None):
  """
  A class decorator that you can use to help define a term algebra, these
  algebras can be run over a context in order to calculate this or that.

  Basically look at `test/test_context.py` for examples on how to use this,
  specifically the `test_algebra_*` functions.

  TODO :: There's a more appropriate way of adding ABC abstract methods to this
         class instead of just hacking in a `__new__`, basically figure out
         how to combine `type()` and `wrapt` to get this working properly. 

  TODO :: Add an option/version of this that allows you to use some specified
         dunder method defined within the term class itself. 
  """

  lang_list = None

  # Get the language definition into the right format 
  if lang == None :
    lang = frozenset([Term])
    lang_list = list(lang)
  elif isinstance(lang, Iterable):
    lang_list = list(lang)
    lang = frozenset(lang_list)
  elif isinstance(alg, Type):
    lang = frozenset([lang])
    lang_list = list(lang)
  else:
    raise TypeError("Could not get language of this algebra.") 

  # Make sure all the terms in the language are reasonable. 
  _validate_lang(lang)

  def _decorate(cls):

    # Add the term algebra as a "parent" class
    _Term_Algebra.register(cls)

    # Build the dictionary of individual reduction functions, and add them to
    # our parent class. 
    run_funcs : Dict[Type,str] = dict()

    for trms in lang:
      f_name = "run_" + trms.__name__.lower()
      run_funcs[trms] = f_name

    cls.language : ClassVar[FrozenSet[Type]] = run_funcs
    
    # Build our full reduction function
    def _run_d(self, ident, val):
      # Get the type in reverse mro order 
      reverse_mros = reversed(getmro(type(val)))
      mros = getmro(type(val))

      # If the reduction functions exist call them.
      # TODO :: Add an option for not calling all of them?
      for ttyp in mros:
        if ttyp in run_funcs:
          if hasattr(self, self.language[ttyp]):
             getattr(self, self.language[ttyp])(ident, val)
             # break out of the for loop on the first valid hit, if we're
             # going in MRO order, this means that only one run run function
             # per item, and it's the most "appropriate" one
             break

    run_defined = False
    if not hasattr(cls, "_run"):
      setattr(cls, "_run", _run_d)
    else:
      run_defined = True


    def _new(cls,*args,**kwargs): 
      missing_funcs = list()
      nonlocal lang_list
      for trms in lang_list: 
        if not hasattr(cls, cls.language[trms]):
          # NOTE :: The reason I'm doing it this way is so that you can just
          #        copy and paste a chunk of the error message into your
          #        algebra and be sorted.
          missing_funcs.append("def " + cls.language[trms] + "(self, ident : 'ID["
                               + trms.__name__ + "]', val : '" + trms.__name__ +
                               "[ID]') -> None:\n        raise "+
                               "NotImplementedError\n ")

      # Throw an error if there is no function with the correct attributes in
      # the class.
      if missing_funcs and (not run_defined):
        raise TypeError("This class must define the following functions:\n\n   "
                        + " \n    ".join(missing_funcs) + "\n\n Copy and " +
                        "paste the above stubs into your algebra class, if " +
                        "needed.")

      return super(cls,cls).__new__(cls)

    setattr(cls, "__new__", _new)
    return cls

  return _decorate


  

