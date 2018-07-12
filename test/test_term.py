from mechsynth.term import *
from typing import *
from pytest import * # type: ignore
from operator import add
from dataclasses import dataclass # type: ignore

@term
class Foo() :
    """
    Just a term we can use in testing whether
    tmap works as intended.
    """
    fst : TermType
    snd : TermType

def test_tmap_basic():
    a = Foo(1,2)
    b = Foo(2,4)
    assert tmap(lambda x : 2*x , a) == b

def test_tmap_complex():
    a = Foo(1,2)
    b = Foo("1","2")
    assert tmap(str, a) == b 

def test_tfold_sum():
    a = Foo(3,4)
    assert tfold(add, 0, a) == 7

def test_tfold_count():
    a = Foo(8,9)
    b = Foo(1,2)
    def cnt(a : int, b : int) -> int :
        return (b + 1)

    assert tfold(cnt, 0, a) == tfold(cnt, 0 ,b)

def test_map_accum_print():
   a = Foo("Hello ","World!")

   def prnlen(a : str, b : str) -> Tuple[str, int]:
     b += a
     l = len(b)
     return (b, l)

   assert map_accum(prnlen,"", a) == ("Hello World!", Foo(6,12))
