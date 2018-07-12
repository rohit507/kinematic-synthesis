from mechsynth.term import *
from mechsynth.context import * # type: ignore
from mechsynth.errors import *
from pytest import *
from typing import *
import pytest 
from dataclasses import dataclass # type: ignore
from pprint import pprint # type: ignore

# We create the terms for a basic abstract syntax tree here, one that
# we can use within a context, so that we can keep track of provenance and
# dependencies.
#
# This will later be very useful as we translate terms from one (high-level)
# language to another. 

@term
class Var() :
    """
    A single variable. With a phantom type (We don't store any values of that
    type but we need to satisfy the term interface.)
    """
    name : str

@term
class Add() :
    """
    The addition operation element in our abstract syntax tree.
    """
    exp_a : TermType
    exp_b : TermType

@term
class Sub() :
    """
    The subtraction operation element in our abstract syntax tree.
    """
    exp_a : TermType
    exp_b : TermType

@term
class Mul() :
    """
    The multiplication operation element in our abstract syntax tree.
    """
    exp_a : TermType
    exp_b : TermType

@term
class Val() :
    """
    A single fixed floating point number.
    """
    val : int

def test_context_equality():
    ctxt = Context("Test")

    a = ctxt.insert(Val(3))
    b = ctxt.insert(Val(3))

    assert a == b

def test_context_inequality():

    ctxt = Context("Test")

    a = ctxt.insert(Val(3))
    b = ctxt.insert(Val(4))

    assert a != b

def test_context_insert_fresh():

    ctxt = Context("Test")

    a = ctxt.insert_fresh(Var("X"))
    b = ctxt.insert_fresh(Var("X"))

    assert a != b

def test_insert_nest():

    ctxt = Context("Test")

    x = ctxt.insert_fresh(Var("X"))
    b = ctxt.insert(Add(x,Mul(x,Val(3))))

    assert len(ctxt.topolist()) == 4


def test_insert_key_store():

    ctxt = Context("Text")

    x = ctxt.insert_fresh(Var('X'))
    y = ctxt.insert_fresh(Var('Y'))

    b = ctxt.insert(Add(x,Mul(y,Val(5))))
    c = ctxt.insert(Add(x,Mul(y,Val(5))))

    b["test"] = "test key"

    assert b == c 

    assert c["test"] == "test key" 

def test_accept_lang():

    # The language of constructors we are allowed to use with this context. 
    lang = frozenset([Var, Add, Sub, Mul, Val])

    ctxt = Context("Foo", _term_language=lang)

    x = ctxt.insert_fresh(Var('X'))
    y = ctxt.insert_fresh(Var('Y'))

    b = ctxt.insert(Add(x,Mul(y,Val(5))))
    c = ctxt.insert(Add(x,Mul(y,Val(5))))

    b["test"] = "test key"

    assert b == c 

    assert c["test"] == "test key" 

def test_reject_lang():

    # The language of constructors we are allowed to use with this context. 
    lang = frozenset([Var, Add, Sub, Mul])

    ctxt = Context("Foo", _term_language=lang)

    x = ctxt.insert_fresh(Var('X'))
    y = ctxt.insert_fresh(Var('Y'))

    with pytest.raises(TermNotInLanguageError): 
      b = ctxt.insert(Add(x,Mul(y,Val(5))))

def test_algebra_decorator_defs():
    """
    Test whether the decorator correctly throws an error when required
    functions are missing. 
    """

    # The language of constructors we are allowed to use with this context. 
    lang = frozenset([Var, Add, Sub, Mul, Val])

    @term_algebra(lang)
    class Foo(): pass

    with pytest.raises(TypeError): 
        printer = Foo()


def test_algebra_print():
    """ 
    Define and test an algebra that acts as a pretty printer.
    """

    # The language of constructors we are allowed to use with this context. 
    lang = frozenset([Var, Add, Sub, Mul, Val])

    @term_algebra(lang)
    class PPrint():
        """
        For every term in our language we write a function that can add some
        info to its key value store.

        Because we traverse in order, this will flesh out those pieces of
        information in a nice bottom-up manner. 
        """

        def _init_algebra(self, ctxt):
            pass
        def _init_pass(self, ctxt):
            pass
        def _end_pass(self, ctxt):
            return False
        def _end_algebra(self, ctxt):
            return None

        def run_add(self, ident : 'ID[Add]', val : 'Add[ID]'):
            ident["pp"] = "(" + val.exp_a["pp"] + " + " + val.exp_b["pp"] + ")"

        def run_sub(self, ident : 'ID[Sub]', val : 'Sub[ID]') -> None:
            ident["pp"] = "(" + val.exp_a["pp"] + " - " + val.exp_b["pp"] + ")"
            
        def run_mul(self, ident : 'ID[Mul]', val : 'Mul[ID]') -> None:
            ident["pp"] = "(" + val.exp_a["pp"] + " * " + val.exp_b["pp"] + ")"

        def run_var(self, ident : 'ID[Var]', val : 'Var[ID]') -> None:
            ident["pp"] = val.name 

        def run_val(self, ident : 'ID[Val]', val : 'Val[ID]') -> None:
            ident["pp"] = repr(val.val) 


    ctxt = Context("Foo", _term_language=lang)

    x = ctxt.insert_fresh(Var('X'))
    y = ctxt.insert_fresh(Var('Y'))

    c = ctxt.insert(Add(x,Mul(y,Val(5))))

    printer = PPrint()

    ctxt.run_algebra(printer)

    assert x["pp"] == "X"
    assert c["pp"] == "(X + (Y * 5))"

def test_algebra_eval():
    """ 
    Define and test an algebra that acts as a simple evaluator.
    """

    # The language of constructors we are allowed to use with this context. 
    lang = frozenset([Add, Sub, Mul, Val])

    @term_algebra(lang)
    class Eval():
        """
        This simple evaluator doesn't support any state.
        """
        def _init_algebra(self, ctxt):
            pass

        def _init_pass(self, ctxt):
            pass

        def _end_pass(self, ctxt):
            return False

        def _end_algebra(self, ctxt):
            return None
    
        def run_add(self, ident : 'ID[Add]', val : 'Add[ID]'):
            ident["res"] = val.exp_a["res"] + val.exp_b["res"]

        def run_sub(self, ident : 'ID[Sub]', val : 'Sub[ID]') -> None:
            ident["res"] = val.exp_a["res"] - val.exp_b["res"]
            
        def run_mul(self, ident : 'ID[Mul]', val : 'Mul[ID]') -> None:
            ident["res"] = val.exp_a["res"] * val.exp_b["res"]

        def run_val(self, ident : 'ID[Val]', val : 'Val[ID]') -> None:
            ident["res"] = val.val


    ctxt = Context("Foo", _term_language=lang)

    c = ctxt.insert(Add(Val(4),Mul(Val(6),Val(5))))

    ctxt.run_algebra(Eval())

    assert c["res"] == 34

def test_algebra_eval_state():
    """ 
    Define and test an algebra that acts as a pretty printer.
    """

    # The language of constructors we are allowed to use with this context. 
    lang = frozenset([Add, Sub, Mul, Val, Var])

    @term_algebra(lang)
    @dataclass
    class Eval():
        """
        This simple evaluator stores variable state 
        """

        variables : Dict[str,int]

        def __init__(self, vs):
            self.variables = vs

        def _init_algebra(self, ctxt):
            pass
        def _init_pass(self, ctxt):
            pass
        def _end_pass(self, ctxt):
            return None
        def _end_algebra(self, ctxt):
            return None
        

        def run_add(self, ident : 'ID[Add]', val : 'Add[ID]'):
            ident["res"] = val.exp_a["res"] + val.exp_b["res"]

        def run_sub(self, ident : 'ID[Sub]', val : 'Sub[ID]') -> None:
            ident["res"] = val.exp_a["res"] - val.exp_b["res"]
            
        def run_mul(self, ident : 'ID[Mul]', val : 'Mul[ID]') -> None:
            ident["res"] = val.exp_a["res"] * val.exp_b["res"]

        def run_val(self, ident : 'ID[Val]', val : 'Val[ID]') -> None:
            ident["res"] = val.val

        def run_var(self, ident : 'ID[Var]', val : 'Var[ID]') -> None:
            if val.name in self.variables:
                ident["res"] = self.variables[val.name]
            else:
                raise ValueError("No variable found.")


    ctxt = Context("Foo", _term_language=lang)
    evaluator = Eval(dict(X=12))

    c = ctxt.insert(Add(Val(4),Mul(Var('X'),Val(5))))

    ctxt.run_algebra(evaluator)

    assert c["res"] == 64

    d = ctxt.insert(Sub(Val(2), Var('Y')))

    with pytest.raises(ValueError):
        ctxt.run_algebra(evaluator)

# TODO :: test_graphviz
# TODO :: test_topo
# TODO :: test_dependency_map
# TODO :: test_topoList
