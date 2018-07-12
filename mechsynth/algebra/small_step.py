from mechsynth.term import *
from mechsynth.context import ID
from mechsynth.symbolic.value import *
from mechsynth.symbolic.geom import *
from mechsynth.symbolic.object import *
from mechsynth.symbolic.relation import *
from mechsynth.symbolic.operator import *
from mechsynth.symbolic.model import *
from mechsynth.symbolic.model_ops import *
from pytest import * # type: ignore
from pprint import *
import pathlib
import math
from enum import Enum, auto 

"""
This module has the small step semantics algebra that tries to reduce all
available terms to versions that are as flat as possible.

In particular, they want to reduce every term to a value type whose non-value
members are functions over single variables and not vectors and stuff.
"""

def final(i):
    if not 'next' in i:
        return i
    elif i['next'] == None:
        return i
    else:
        final(i['next'])

class TType(Enum):
    UNK = auto()     # We don't know yet what type of expression this is 
    SCALAR = auto()  # An expression over only
    VECTOR = auto()  # A vector of scalar terms
    MATRIX = auto()  # A vector of vectors

@model_algebra
class TTypeAlg():
    """
    Figures out the tensor type of each term in our language, if we can tell.
    This lets us ensure that all of our final terms have the proper tensor
    type. 
    """
    def _init_algebra(self, ctxt):
        ctxt.purge_keys({'tensor_type'})

    def _init_pass(self, ctxt):
        pass

    def _end_pass(self, ctxt):
        return False

    def _end_algebra(self, ctxt):
        return None

    def _run(self, ident, val) -> None:
        """
        We override the run function because it's basically the same code
        except for the three core classes.
        """


        if 'tensor_type' in ident:
            return None

        ttype = TType.UNK

        # If we're something interesting, we use ourselves.
        if (type(val) in {Param, Constant, Control, Variable}):
            # Core Types : We can't reduce these any further , but the are
            # interpretable by dreal and stuff. 
            ttype = TType.SCALAR
        elif (type(val) in {Mat3x1,Mat4x1}):
            # If every sub element is a scalar expression then this is a vector,
            # otherwise unknown

            tl = tlist(val)

            for i in tl: 
                self._run(i, i.val)

            if all(e['tensor_type'] == TType.SCALAR for e in tl):
                ttype = TType.VECTOR
        elif (type(val) in {Point, Vector, Unit_Vector}):

            # Points, vectors, and unit vectors require a well formed frame in
            # order to be considered well formed themselves. 


            for i in [val.x, val.y, val.z, val.w, val.parent_frame]:  
                self._run(i, i.val)

            cx = val.x['tensor_type'] == TType.SCALAR
            cy = val.y['tensor_type'] == TType.SCALAR
            cz = val.z['tensor_type'] == TType.SCALAR
            cw = val.w['tensor_type'] == TType.SCALAR
            cf = val.parent_frame['tensor_type'] == TType.MATRIX

            if all([cx, cy, cz, cw, cf]):
                ttype = TType.VECTOR

        elif (type(val) in {Anchor}):

            # Points, vectors, and unit vectors require a well formed frame in
            # order to be considered well formed themselves. 


            for i in [val.x, val.y, val.z, val.w, val.parent_frame,
                      val.exists, val.direction, val.handle]:  
                self._run(i, i.val)

            cx = val.x['tensor_type'] == TType.SCALAR
            cy = val.y['tensor_type'] == TType.SCALAR
            cz = val.z['tensor_type'] == TType.SCALAR
            cw = val.w['tensor_type'] == TType.SCALAR

            cf = val.parent_frame['tensor_type'] == TType.MATRIX

            ce = val.exists['tensor_type'] == TType.SCALAR

            cd = val.direction['tensor_type'] == TType.VECTOR
            ch = val.handle['tensor_type'] == TType.VECTOR

            if all([cx, cy, cz, cw, cf, ce, cd, ch]):
                ttype = TType.VECTOR

        elif (type(val) in {Line}):

            for i in [val.direction, val.moment]:
                self._run(i, i.val)

            c_d = val.direction['tensor_type'] == TType.VECTOR
            c_m = val.moment['tensor_type'] == TType.VECTOR

            if all([c_d, c_m]):
                ttype = TType.VECTOR
            
        elif (type(val) in [Mat3x3,Mat4x4,SOMat3x3, SEMat4x4,
                                         Frame]):

            tl = tlist(val)

            for i in tl: 
                self._run(i, i.val)

            # if every element is a vector, then this is a matrix.
            if all(e['tensor_type'] == TType.VECTOR for e in tl):
                ttype = TType.MATRIX

        elif (type(val) == Body):

            for i in [val.c1, val.c2, val.c3, val.c4,
                      val.exists]:
                self._run(i, i.val)
            c_c1 = val.c1['tensor_type'] == TType.VECTOR
            c_c2 = val.c2['tensor_type'] == TType.VECTOR
            c_c3 = val.c3['tensor_type'] == TType.VECTOR
            c_c4 = val.c4['tensor_type'] == TType.VECTOR

            c_e = val.exists['tensor_type'] == TType.SCALAR

            if all([c_c1, c_c2, c_c3, c_c4, c_e]):
                ttype = TType.MATRIX

        elif (type(val) == FrameTransform):

            for i in [val.c1, val.c2, val.c3, val.c4,
                      val.parent_frame, val.target_frame]:
                self._run(i, i.val)

            c_c1 = val.c1['tensor_type'] == TType.VECTOR
            c_c2 = val.c2['tensor_type'] == TType.VECTOR
            c_c3 = val.c3['tensor_type'] == TType.VECTOR
            c_c4 = val.c4['tensor_type'] == TType.VECTOR

            c_pf = val.parent_frame['tensor_type'] == TType.MATRIX
            c_tf = val.target_frame['tensor_type'] == TType.MATRIX

            if all([c_c1, c_c2, c_c3, c_c4, c_pf, c_tf]):
                ttype = TType.MATRIX

        elif (type(val) == Hinge):

            for i in [val.c1, val.c2, val.c3, val.c4,
                      val.parent_frame, val.target_frame,
                      val.exists, val.axis, val.angle,
                      val.min_angle, val.max_angle]:
                self._run(i, i.val)

            c_c1 = val.c1['tensor_type'] == TType.VECTOR
            c_c2 = val.c2['tensor_type'] == TType.VECTOR
            c_c3 = val.c3['tensor_type'] == TType.VECTOR
            c_c4 = val.c4['tensor_type'] == TType.VECTOR

            c_pf = val.parent_frame['tensor_type'] == TType.MATRIX
            c_tf = val.target_frame['tensor_type'] == TType.MATRIX

            c_e = val.exists['tensor_type'] == TType.SCALAR

            c_ax = val.axis['tensor_type'] == TType.VECTOR
            c_an = val.angle['tensor_type'] == TType.SCALAR
            c_na = val.min_angle['tensor_type'] == TType.SCALAR
            c_xa = val.max_angle['tensor_type'] == TType.SCALAR

            if all([c_c1, c_c2, c_c3, c_c4, c_pf, c_tf, c_e, c_ax, c_an,
                    c_na, c_xa]):
                ttype = TType.MATRIX


        elif (type(val) == Slide):

            for i in [val.c1, val.c2, val.c3, val.c4,
                      val.parent_frame, val.target_frame,
                      val.exists, val.axis, val.offset,
                      val.min_offset, val.max_offset]:
                self._run(i, i.val)

            c_c1 = val.c1['tensor_type'] == TType.VECTOR
            c_c2 = val.c2['tensor_type'] == TType.VECTOR
            c_c3 = val.c3['tensor_type'] == TType.VECTOR
            c_c4 = val.c4['tensor_type'] == TType.VECTOR

            c_pf = val.parent_frame['tensor_type'] == TType.MATRIX
            c_tf = val.target_frame['tensor_type'] == TType.MATRIX

            c_e = val.exists['tensor_type'] == TType.SCALAR

            c_ax = val.axis['tensor_type'] == TType.VECTOR
            c_of = val.offset['tensor_type'] == TType.SCALAR
            c_no = val.min_offset['tensor_type'] == TType.SCALAR
            c_xo = val.max_offset['tensor_type'] == TType.SCALAR

            if all([c_c1, c_c2, c_c3, c_c4, c_pf, c_tf, c_e, c_ax, c_of,
                    c_no, c_xo]):
                ttype = TType.MATRIX


        elif (type(val) in {Mul, Add, Pow, Negate, 
                                         Eq, IfThenElse, And, Or, Not, Xor,
                                         Implies, Mag,
                                         LessThan, LessThanEq,
                                         GreaterThan, GreaterThanEq,
                                         Sin, Cos, Tan,
                                         Asin, Acos, Atan, Atan2}):
            # Should be everything else, if the subexpression is scalar, then
            # the result is scalar, otherwise unknown. 
            tl = tlist(val)

            for i in tl: 
                self._run(i, i.val)

            if all(e['tensor_type'] == TType.SCALAR for e in tl):
                ttype = TType.SCALAR
        elif (type(val) in {Dot, GetMember, AtInitial, Invert,
                                         Sub, Div, Sqrt,
                                         Between, Within, Cross,
                                         CrossMat,  MatMul, 
                                         Norm, Dist, Line, Hinge,
                                         Anchor, FrameTransform, 
                                         InFrame, }):
            # These are terms which shouldn't be in our fully reduced algebra
            # Terms that dreal can't understand on its own. 
            ttype = TType.UNK
        else:
            raise TypeError(
                f' `{type(val)}` not found in tensor type algebra.')


        ident['tensor_type'] = ttype


@model_algebra
class FinalizeAlg():
    """
    An algebra that updates all the available terms to versions that are
    based around the final constructions of each of their terms.

    This means that other passes can just lookup t.final for the fully
    reduced versions of all the terms. 
    """

    dirty : bool = False

    def _init_algebra(self, ctxt):
        pass

    def _init_pass(self, ctxt):
        self.dirty = False

    def _end_pass(self, ctxt):
        return self.dirty

    def _end_algebra(self, ctxt):
        return None

    def _run(self, ident, val) -> None:
        ident = id_to_mval(ident)
        if not ident.has_next:

            def update_term(ident, dirty):
                n = ident.final
                if n != ident: dirty = True
                return (dirty, n)

            dirty, new = map_accum(update_term, False, val)

            if dirty:
                ident.set_next(new, color='deeppink4:deepskyblue4')
                self.dirty = True


        
@model_algebra
@dataclass
class SmallStepAlg():
    """
    Basically, we want to define what each symbol in our language means, by
    defining how we transform it into a 'simpler' statement.

    Our definition of simpler basically means 'within DReal's sublanguage'.

    We use the 'next' flag to store pointers to the relevant term.
    """

    dirty : bool = False
    tt_alg : TTypeAlg = field(default_factory=TTypeAlg)
    counter : int = 1
    max_runs : int = 5
    debug : bool = False

    def _init_algebra(self, ctxt):
        ctxt.purge_keys({'untouched'})
        pass

    def _init_pass(self, ctxt):
        self.dirty = False

        #print(f'Running small step reduction pass #{self.counter}')

        self.counter += 1
        # Run the tensor type algebra at the start of each pass. 
        ctxt.run_algebra(TTypeAlg())
        pass

    def _end_pass(self, ctxt):

        falg = FinalizeAlg()
        ctxt.run_algebra(falg)
        self.dirty = self.dirty or falg.dirty

        return self.dirty and self.counter < self.max_runs

    def _end_algebra(self, ctxt):
        return None

    def _run(self, ident : MVal, val : Term) -> None:
        if type(ident) == ID:
            ident = id_to_mval(ident)

        ## Run the Tensortype algebra on the term we're viewing now. 
        self.tt_alg._run(ident, val) 

        if not ident.has_next:


            if (type(val) in {Param, Constant, Control, Variable}):
                ## These are final elements as per our small step reduction,
                ## we can't actually reduce these expressions further. 
                pass
            elif (type(val) in {Mat4x1, Mat3x1, Mat4x4, Mat3x3,
                                             Vector, Point, SOMat3x3, SEMat4x4,
                                             Unit_Vector, Frame, FrameTransform,
                                             Line, Body, Anchor, Hinge}):
                # these are not terms that dreal supports directly, we will
                # need to decompose them into flat expressions

                # basically we shouldn't have to worry about them. Between
                # external expressions collapsing around them, and their own
                # internal representations being simplified, these should
                # more or less just be extra metadata for organizational
                # purposes w/ no bearing on the actual result. 
                pass
            elif (type(val) in {LessThan, GreaterThan,
                                             LessThanEq, GreaterThanEq,
                                             And, Or, Xor, Not, Implies,
                                             Sin, Cos, Tan, Pow,
                                             Asin, Acos, Atan, Atan2}):
                # These terms can't really be distributed into leaves, and are
                # otherwise terminals. 
                if not all(((t['tensor_type'] == TType.SCALAR) or
                        (t['tensor_type'] == TType.UNK)) for t in tlist(val)):
                    raise Exception(f'Term of type {type(val)} contains ' +
                                    'non-scalar elements.')

                # if we are a scalar expression then this is terminal. 
                if all((t['tensor_type'] == TType.SCALAR) for t in tlist(val)):
                    pass
                else:
                    if hasattr(self, self.language[type(val)]):
                        #print(f'small stepping : {type(val).__name__}') 
                        getattr(self, self.language[type(val)])(ident, val)
            elif (type(val) in {IfThenElse, Dot, Eq, Mul, Add, Sub,
                               Div, MatMul, Within, GetMember,
                               InFrame, Negate, 
                               AtInitial, Between, Invert, Mag,
                               Cross, Sqrt, Norm, Dist, CrossMat}):
                # Pass these things to the corresponding run function. 
                if hasattr(self, self.language[type(val)]):
                    #print(f'small stepping : {type(val).__name__}') 
                    getattr(self, self.language[type(val)])(ident, val)
                else:
                    raise TypeError(f' `{type(val)}` not found in small ' +
                                'step algebra.')

            else:
                if 'untouched' in ident:
                    ident['untouched'] = ident['untouched'] + 1
                else:
                    ident['untouched'] = 1
                #print(f'  Untouched ({ident["untouched"]}): {ident.long_name}')


        # Run the tensor type algebra to make sure all new elements are updated
        # FIXME :: Gods this is hacky and awful. 
        # ident.parent_context.run_algebra(TTypeAlg())

    ### Distributable terms ###

    # We can take a well tensor-typed parameter and create a well tensortyped
    # next statement. Pushing the function down through the setup. 

    def run_ifthenelse(self, ident : 'ID[IfThenElse]', val : 'IfThenElse[ID]') -> None:
        """
        If we've reduced the true and false expressions in this element into
        well tensor typed elements, then we can distribute the conditional
        into the leaves.
        """


        tc = val.exp_cond.final 
        tt = val.exp_true.final 
        tf = val.exp_false.final

        for i in [tc,tt,tf]: 
            self.tt_alg._run(i, i.val) 

        def wrap_if(vt,vf):
           return IfThenElse(tc,vt,vf)

        if all(t['tensor_type'] == TType.SCALAR for t in [tt,tf]):
            # This expression is already as flat as it can get. 
            pass
        elif all(t['tensor_type'] == TType.VECTOR for t in [tt,tf]):
            # Distribute the terms down into the vector. 
            ident.set_next(tzipwith(wrap_if,tt.val, tf.val))
            self.dirty = True
        elif all(t['tensor_type'] == TType.MATRIX for t in [tt,tf]):
            # Distribute the terms down into the vector. 
            ident.set_next(tzipwith(wrap_if,tt.val, tf.val))
            self.dirty = True
        elif all(t['tensor_type'] != TType.UNK for t in [tt,tf]):
            raise Exception("If then else contains elements w/ non-matching" +
                            " tensor type.")

    def run_dot(self, ident : 'ID[Dot]', val : 'Dot[ID]') -> None:
        """
        Convert the vector dot product into a flat expression.
        """

        ta = val.exp_a.final
        tb = val.exp_b.final

        for i in [ta,tb]: 
            self.tt_alg._run(i, i.val) 

        va = ta.val
        vb = tb.val

        if all(t['tensor_type'] == TType.VECTOR for t in [ta,tb]):

            if all(type(v) == Mat3x1 for v in [va,vb]) :
                # This is a simple column vector, we can construct the dot
                # product manually
                ident.set_next((va.x * vb.x) + (va.y * vb.y) + (va.z * vb.z))
                self.dirty = True
            elif all((type(v) in {Mat4x1, Vector, Unit_Vector, Point}) for v in [va,vb]):
                # Dot prodcuts are much the same for vectors and unit-vectors.
                # note how we exclude the dot products of points, since those
                # are meaningless. 
                ident.set_next((va.x * vb.x) + (va.y * vb.y)
                               + (va.z * vb.z) + (va.w * vb.w))
                self.dirty = True
            else:
                print("no dot")
                print(va)
                print(vb)
                raise NotImplementedError

        elif all(t['tensor_type'] != TType.UNK for t in [ta,tb]):
            raise Exception("can't take dot product of invalid types.")


    def run_eq(self, ident : 'ID[Eq]', val : 'Eq[ID]') -> None:

        ta = val.exp_a.final
        tb = val.exp_b.final

        for i in [ta,tb]: 
            self.tt_alg._run(i, i.val)

        va = ta.val
        vb = tb.val

        if ta == tb:
            print("eliminating known Eq")
            # if you are pointing to literally the same object you m
            ident.set_next(Constant(True), color='forestgreen:gold1:forestgreen')
        elif ('eq_set' in ta) and (tb in ta['eq_set']):
            pass
            #print("eliminating known Eq")
            # likewise, if you are talking about things which are known to be
            # equal ... then we're good. 
            #ident.set_next(Constant(True), color='forestgreen:gold1:forestgreen')
        elif all(t['tensor_type'] == TType.SCALAR for t in [ta,tb]):
            pass # We can't push this down further. 
        elif (ta['tensor_type'] != TType.UNK) and ta['tensor_type'] == tb['tensor_type']:
            # If we've got a pair or vectors or matrices we can distribute the
            # equality term down as needed.\
            #
            # NOTE :: because we've made tzip rather restrictive, I've had to
            #        special case out those elements that are largely identical.
            #        and which should be comparable. 
            vn = None
            if type(va) == type(vb): 
                vn = tzipwith(Eq, va, vb)
            elif all([type(v) in {Mat3x3,SOMat3x3} for v in [va,vb]]):
                vn = Mat3x3(c1= Eq(va.c1, vb.c1),
                            c2= Eq(va.c2, vb.c2),
                            c3= Eq(va.c3, vb.c3))
            elif all([type(v) in {Mat4x4,SEMat4x4,Frame,Body,
                                  Hinge,Slide,FrameTransform} for v in [va,vb]]):
                vn = Mat4x4(c1= Eq(va.c1, vb.c1),
                            c2= Eq(va.c2, vb.c2),
                            c3= Eq(va.c3, vb.c3),
                            c4= Eq(va.c4, vb.c4))
            elif all([type(v) in {Mat4x1,Point,Vector,Unit_Vector, Anchor} for v in [va,vb]]):
                vn = Mat4x1(x = Eq(va.x, vb.x),
                            y = Eq(va.y, vb.y),
                            z = Eq(va.z, vb.z),
                            w = Eq(va.w, vb.w))
            elif all([type(v) in {Mat3x1,Point, Anchor} for v in [va,vb]]):
                vn = Mat3x1(x = Eq(va.x, vb.x),
                            y = Eq(va.y, vb.y),
                            z = Eq(va.z, vb.z))
            elif all([type(v) in {Mat3x1,Vector,Unit_Vector,Mat4x1} for v in [va,vb]]):
                vn = Mat3x1(x = Eq(va.x, vb.x),
                            y = Eq(va.y, vb.y),
                            z = Eq(va.z, vb.z))
            else:
                print("Missing EQ case:")
                print()
                print(va)
                print()
                print(vb)
                print()
                raise NotImplementedError
            vf = tfold(And, True, vn) 
            ident.set_next(vf)
            # print(va)
            # print(vb)
            # print(vf)
            # print()
            self.dirty = True

        elif all(t['tensor_type'] != TType.UNK for t in [ta,tb]):
            raise Exception("can't take equality of non_matched types.")

    def run_negate(self, ident : 'ID[Negate]', val : 'Negate[ID]') -> None:

        t = val.exp.final

        for i in [t]: 
            self.tt_alg._run(i, i.val)

        v = t.val

        if (t['tensor_type'] == TType.SCALAR):
            pass
        elif (t['tensor_type'] != TType.UNK):
            # If we've got a pair or vectors or matrices we can distribute the
            # equality term down as needed.

            # Need to do this for
            # Matrices, Frames, vector and unit_vector
            if type(v) in [Mat3x1, Mat4x1, Mat3x3, Mat4x4, SOMat3x3,
                           SEMat4x4, Frame]:
                def neg_elem(v): return v.negation

                ident.set_next(tmap(neg_elem, v))
                self.dirty = True
            elif type(v) == Vector:
                ident.set_next(Vector(x = -v.x,
                                      y = -v.y,
                                      z = -v.z,
                                      w = -v.w,
                                      parent_frame = v.parent_frame))
                self.dirty = True
            elif type(v) == Unit_Vector:
                ident.set_next(Unit_Vector(x = -v.x,
                                      y = -v.y,
                                      z = -v.z,
                                      w = -v.w,
                                      parent_frame = v.parent_frame))
                self.dirty = True
            # elif type(v) == Point:
            #     ident.set_next(Point(x = -v.x,
            #                           y = -v.y,
            #                           z = -v.z,
            #                           w = 1.0,
            #                           parent_frame = v.parent_frame))
            #     self.dirty = True
            else:
                print("no negate")
                print(v)
                raise NotImplementedError

        elif (t['tensor_type'] == TType.UNK):
            pass
        else:
            raise Exception("Can't take negation of given type.") 
                                    


    def run_mul(self, ident : 'ID[Mul]', val : 'Mul[ID]') -> None:
        """
        We can distribute scalar * vector, or scalar * matrix multiplication
        down
        """

        ta = val.exp_a.final
        tb = val.exp_b.final

        for i in [ta,tb]: 
            self.tt_alg._run(i, i.val)

        if (tb['tensor_type'] == TType.SCALAR
            and (ta['tensor_type'] not in [TType.UNK, TType.SCALAR])):
            # Swap so that a scalar (if we have it) is in ta
            ta, tb = tb, ta 

        va = ta.val
        vb = tb.val

        if(ta['tensor_type'] == TType.SCALAR and tb['tensor_type'] == TType.VECTOR):

            if type(vb) in [Mat3x1,Mat4x1]: 
                ident.set_next(tmap(lambda t: Mul(ta,t), vb))
                self.dirty = True
            elif type(vb) in [Vector,Unit_Vector]:
                ident.set_next(Vector(x = Mul(va,vb.x),
                                      y = Mul(va,vb.y),
                                      z = Mul(va,vb.z),
                                      w = 0.0,
                                      parent_frame = vb.parent_frame))
                self.dirty = True
            else:
                print(va)
                print(vb)
                raise NotImplementedError
        
        elif(ta['tensor_type'] == TType.SCALAR and tb['tensor_type'] == TType.MATRIX):

            if type(vb) in [Mat3x3,Mat4x4]:
                ident.set_next(tmap(lambda t: Mul(ta,t), vb))
                self.dirty = True
            else:
                print("no scalar matrix mul")
                print(va)
                print(vb)
                raise NotImplementedError

        elif(ta['tensor_type'] == TType.SCALAR and tb['tensor_type'] == TType.SCALAR):

            pass

        elif all(t['tensor_type'] != TType.UNK for t in [ta,tb]):
            raise Exception("Can't multiply non-scalar types")

    def run_div(self, ident : 'ID[Div]', val : 'Div[ID]') -> None:
        """
        We can convert (a / b) into (a * (1/b)).
        """

        ta = val.exp_a.final
        tb = val.exp_b.final

        for i in [ta,tb]: 
            self.tt_alg._run(i, i.val)

        ident.set_next(Mul(ta,Pow(tb,Constant(-1.0))))
        self.dirty = True

    def run_add(self, ident : 'ID[Add]', val : 'Add[ID]') -> None:
        """
        We can distribute scalar + vector, or scalar + matrix multiplication
        down

        TODO :: Add cases for points and vectors, such that the following are
               allowed, and the frames must be equal. (do we add a constraint
               directly? or just have some other mechanism?, try an
               explicit equality check first.) 

               P - P = V (?) 
               P + V = P
               V + V = V

        """

        ta = val.exp_a.final
        tb = val.exp_b.final

        for i in [ta,tb]: 
            self.tt_alg._run(i, i.val)

        va = ta.val
        vb = tb.val

        if(ta['tensor_type'] == TType.SCALAR and tb['tensor_type'] == TType.SCALAR):
            pass
        elif(ta['tensor_type'] == TType.VECTOR and tb['tensor_type'] == TType.VECTOR):
            # Cases
            if((type(va) == Mat3x1) and (type(vb) == Mat3x1)):
                ident.set_next(Mat3x1(va.x + vb.x,
                                      va.y + vb.y,
                                      va.z + vb.z))
                self.dirty = True        
            elif((type(va) == Mat4x1) and (type(vb) == Mat4x1)):
                ident.set_next(Mat4x1(va.x + vb.x,
                                      va.y + vb.y,
                                      va.z + vb.z,
                                      va.w + vb.w))
                self.dirty = True        
            elif((type(va) == Point) and (type(vb) == Vector)):
                ident.set_next(Point(x = va.x + vb.x,
                                     y = va.y + vb.y,
                                     z = va.z + vb.z,
                                     w = va.w + vb.w,
                                     parent_frame = va.parent_frame))
                self.dirty = True        
            elif((type(va) == Vector) and (type(vb) == Point)):
                ident.set_next(Point(x = va.x + vb.x,
                                     y = va.y + vb.y,
                                     z = va.z + vb.z,
                                     w = va.w + vb.w,
                                     parent_frame = va.parent_frame))
                self.dirty = True        
            elif all([type(v) in {Vector, Unit_Vector} for v in [va, vb]]):
                ident.set_next(Vector(x = va.x + vb.x,
                                     y = va.y + vb.y,
                                     z = va.z + vb.z,
                                     w = va.w + vb.w, 
                                     parent_frame = va.parent_frame))
                self.dirty = True
            else:
                print("no vector sum")
                print(va)
                print(vb)
                raise NotImplementedError

        elif(ta['tensor_type'] == TType.MATRIX and tb['tensor_type'] == TType.MATRIX):

            if all([type(v) in [Mat3x3, SOMat3x3] for v in [va,vb]]):
                ident.set_next(Mat3x3(c1 = va.c1 + vb.c1,
                                      c2 = va.c2 + vb.c2,
                                      c3 = va.c3 + vb.c3))

            elif all([type(v) in [Mat4x4, SEMat4x4] for v in [va,vb]]):
                ident.set_next(Mat4x4(c1 = va.c1 + vb.c1,
                                      c2 = va.c2 + vb.c2,
                                      c3 = va.c3 + vb.c3,
                                      c4 = va.c4 + vb.c4))
                self.dirty = True        
            else:
                print("no matrix sum")
                print(va)
                print(vb)
                raise NotImplementedError
            # NOTE :: We shouldn't be adding rotation matrices or frames ever
            #        that doesn't make sense. We should see how well this
            #        functions at runtime. 

        elif all(t['tensor_type'] != TType.UNK for t in [ta,tb]):
            raise Exception(f'Can\'t add types {ta} and {tb}')

    def run_sub(self, ident : 'ID[Add]', val : 'Add[ID]') -> None:
        """
        We can convert (a - b) into (a + (-b)).
        """

        ta = val.exp_a.final
        tb = val.exp_b.final

        for i in [ta,tb]: 
            self.tt_alg._run(i, i.val)

        ident.set_next(Add(ta,Negate(tb)))
        self.dirty = True

    def run_sqrt(self, ident : 'ID[Sqrt]', val : 'Sqrt[ID]') -> None:
        """
        We can convert sqrt(a) into pow(a, 0.5)
        """

        t = val.exp.final
        v = t.val

        for i in [t]: 
            self.tt_alg._run(i, i.val)

        ident.set_next(Pow(t,Constant(0.5)))
        self.dirty = True

    def run_within(self, ident : 'ID[Within]', val : 'Within[ID]') -> None:
        """
        a.within(dist=d, of=o) is equivalent to (d >= Dist(a, b))
        """

        tv = val.exp_val.final 
        td = val.exp_dist.final 
        to = val.exp_of.final

        for i in [tv,td,to]: 
            self.tt_alg._run(i, i.val)

        ident.set_next(td >= tv.dist_to(to))
        self.dirty = True

    def run_between(self, ident : 'ID[Between]', val : 'Between[ID]') -> None:
        """
        a.between(min, max) == a >= min && <= Max 
        """

        tv = val.exp_val.final 
        tn = val.exp_min.final 
        tx = val.exp_max.final

        for i in [tv,tn,tx]: 
            self.tt_alg._run(i, i.val)

        ident.set_next((tv >= tn) & (tv <= tx))
        self.dirty = True

    def run_atinitial(self, ident : 'ID[AtInitial]', val : 'AtInitial[ID]') -> None:
        """
        Get the given expression at some initial conditions. 
        """

        tz = val.exp.final

        for i in [tz]:
            self.tt_alg._run(i, i.val)

        vz = tz.val

        col = 'orangered:orchid:orangered'

        if type(vz) in {Control, Variable}:
            # If we're working with a control or variable, we can just get the
            # initial condition. 
            ident.set_next(vz.initial_condition, color=col)
        elif type(vz) in {Param, Constant, AtInitial}:
            ident.set_next(vz, color=col)
        else:
            # NOTE :: This is if there are any terms within, things like
            #        constants should just 
            # Otherwise, we can just push the initial condition value down into
            # each of the terms as needed.
            # Even being generic is fine here, since it's a standard
            # replacement.
            
            ident.set_next(tmap(lambda t: AtInitial(t), vz), color=col)

        self.dirty = True


    def run_getmember(self, ident, val) -> None:
        """
        When we can decompose terms, we do. Wait till the expression is
        flattened enough to make it possible for us to grab specific terms
        we require.

        We just use these to grab elements out of other matrices when the
        reductions are done. There's going to be a lot of gratuitous entries
        of this sort as we flow through our reduction, since it lets us
        describe a lot of the relevant relationships much more easily. 

        some special elements like the row matrices are held separately

        NOTE :: We use orange as the next_col here because it lets us more
               easily trace when these are removed. 
        """

        n = val.member
        t = val.exp.final

        for i in [t]: 
            self.tt_alg._run(i, i.val)

        v = t.val

        ## Some members are special cased in
        if n == "r1":
            if type(v) in [Mat3x3,SOMat3x3]:
                ident.set_next(Mat3x1(v.c1.x, v.c2.x, v.c3.x), color="orange")
                self.dirty = True
            elif type(v) in [Mat4x4, Frame, FrameTransform, SEMat4x4, Body, Hinge, Slide]:
                ident.set_next(Mat4x1(v.c1.x, v.c2.x, v.c3.x, v.c4.x), color="orange")
                self.dirty = True
            elif t['tensor_type'] != TType.UNK:
                raise Exception("There is no r1 for this type of element.")
        elif n == "r2":
            if type(v) in [Mat3x3,SOMat3x3]:
                ident.set_next(Mat3x1(v.c1.y, v.c2.y, v.c3.y), color="orange")
                self.dirty = True
            elif type(v) in [Mat4x4, Frame, FrameTransform, SEMat4x4, Body, Hinge, Slide]:
                ident.set_next(Mat4x1(v.c1.y, v.c2.y, v.c3.y, v.c4.y), color="orange")
                self.dirty = True
            elif t['tensor_type'] != TType.UNK:
                raise Exception("There is no r2 for this type of element.")
        elif n == "r3":
            if type(v) in [Mat3x3,SOMat3x3]:
                ident.set_next(Mat3x1(v.c1.z, v.c2.z, v.c3.z), color="orange")
                self.dirty = True
            elif type(v) in [Mat4x4, Frame, FrameTransform, SEMat4x4, Body, Hinge, Slide]:
                ident.set_next(Mat4x1(v.c1.z, v.c2.z, v.c3.z, v.c4.z), color="orange")
                self.dirty = True
            elif t['tensor_type'] != TType.UNK:
                raise Exception("There is no r3 for this type of element.")
        elif n == "r4":
            if type(v) in [Mat4x4, Frame, FrameTransform, SEMat4x4, Body, Hinge, Slide]:
                ident.set_next(Mat4x1(v.c1.w, v.c2.w, v.c3.w, v.c4.w), color="orange")
                self.dirty = True
            elif t['tensor_type'] != TType.UNK:
                raise Exception("There is no r4 for this type of element.")
        elif n == "point_offset":
            raise NotImplementedError
        elif n == "rotation_matrix":
            raise NotImplementedError
        elif hasattr(v,n):
            member = getattr(v,n)
            if isinstance(member, ID): 
                ident.set_next(member, color="orange")
                self.dirty = True
            else:
                raise Exception("Trying to access non-term member of term");
        elif t['tensor_type'] != TType.UNK:
            raise Exception("This is a terminal element, but it doesn't have "
                            + "the expected term within it."
                            + f'\n\n{t}\n\n{n}' )
   
    def run_invert(self, ident : 'Invert[Sqrt]', val : 'Invert[ID]') -> None:
        """
        We should only be able to invert the few types of rigid body transforms
        we use, more general matrix inversion is beyond the chose of this tool.

        FIXME :: Yes, this is super cludgy. It has us composing and decomposing
                elements rather a lot and will probably cause a lot of stupid
                overhead.
                part of the reason why we wait till this is a fully
                reduced matrix type is so that the local lookups to each of
                the elements will shortcircuit, instead of requiring 
        """

        t = val.exp.final

        for i in [t]:
            self.tt_alg._run(i, i.val)

        v = t.val

        if (t['tensor_type'] == TType.MATRIX):
            point = Mat3x1(v.c4.x, v.c4.y, v.c4.z)
            rot_mat = Mat3x3(Mat3x1(v.c1.x, v.c1.y, v.c1.z),
                             Mat3x1(v.c2.x, v.c2.y, v.c2.z),
                             Mat3x1(v.c3.x, v.c3.y, v.c3.z))


            rt = Mat3x3(c1 = Mat3x1(v.c1.x, v.c2.x, v.c3.x),
                        c2 = Mat3x1(v.c1.y, v.c2.y, v.c3.y),
                        c3 = Mat3x1(v.c1.z, v.c2.z, v.c3.z))

            np = ident.parent_context.insert(MatMul(Negate(rt),point))
         
            if type(v) in {FrameTransform,Hinge, Slide}:

               ident.set_next(FrameTransform(c1 = Mat4x1(rt.c1.x, rt.c1.y, rt.c1.z, 0.0),
                                             c2 = Mat4x1(rt.c2.x, rt.c2.y, rt.c2.z, 0.0),
                                             c3 = Mat4x1(rt.c3.x, rt.c3.y, rt.c3.z, 0.0),
                                             c4 = Mat4x1(np.x, np.y, np.z, 1.0),
                                             parent_frame=v.target_frame,
                                             target_frame=v.parent_frame))
            elif type(v) in {Frame,Body}:

                def_frame = Frame(c1 = Mat4x1(1.0,0.0,0.0,0.0),
                                  c2 = Mat4x1(0.0,1.0,0.0,0.0),
                                  c3 = Mat4x1(0.0,0.0,1.0,0.0),
                                  c4 = Mat4x1(0.0,0.0,0.0,1.0))

                ident.set_next(FrameTransform(c1 = Mat4x1(rt.c1.x, rt.c1.y, rt.c1.z, 0.0),
                                              c2 = Mat4x1(rt.c2.x, rt.c2.y, rt.c2.z, 0.0),
                                              c3 = Mat4x1(rt.c3.x, rt.c3.y, rt.c3.z, 0.0),
                                              c4 = Mat4x1(np.x, np.y, np.z, 1.0),
                                              parent_frame=def_frame,
                                              target_frame=v))
            elif type(v) == SEMat4x4:
                ident.set_next(SEMat4x4(Mat4x1(rt.c1.x, rt.c1.y, rt.c1.z, 0.0),
                                        Mat4x1(rt.c2.x, rt.c2.y, rt.c2.z, 0.0),
                                        Mat4x1(rt.c3.x, rt.c3.y, rt.c3.z, 0.0),
                                        Mat4x1(np.x, np.y, np.z, 1.0)))
            elif type(v) == SOMat3x3:
                ident.set_next(Transpose(v))
            else:
                print("no matrix inv")
                print(v)
                raise NotImplementedError

            self.dirty = True

        elif (t['tensor_type'] == TType.UNK):
            pass
        else:
            raise Exception("Can't take inverse of non-matrix type.") 
        
    def run_mag(self, ident : 'ID[Mag]', val : 'Mag[ID]') -> None:
        """
        The magnitude of a scalar is a terminal absolute value function,
        the magnitude of a vector is sqrt(dot(v,v)) 
        """

        t = val.exp.final

        for i in [t]:
            self.tt_alg._run(i, i.val)

        v = t.val

        if(t['tensor_type'] == TType.SCALAR):
            pass
        elif(t['tensor_type'] == TType.VECTOR):
            ident.set_next(Pow(Dot(v,v),0.5))
            self.dirty = True
        elif (t['tensor_type'] == TType.UNK):
            pass
        else:
            raise Exception("Can't take inverse of non-matrix type.") 

    def run_cross(self, ident : 'ID[Cross]', val : 'Cross[ID]') -> None:
        """
        We expand out the cross product into the flatter representation

        There's a bunch of cases so that we can propagate the local reference
        frame out more usefully. 

        TODO :: Consider just using matrix multiplication to run this, reducing
               (hopefully) unneccesary code duplication 
        """

        ta = val.exp_a.final
        tb = val.exp_b.final

        for i in [ta,tb]:
            self.tt_alg._run(i, i.val)

        va = ta.val
        vb = tb.val

        if(ta['tensor_type'] == TType.VECTOR and tb['tensor_type'] == TType.VECTOR):

            if((type(va) == Mat3x1) and (type(vb) == Mat3x1)):
                ident.set_next(Mat3x1(x = (va.y * vb.z) - (va.z * vb.y),
                                      y = (va.z * vb.x) - (va.x * vb.z),
                                      z = (va.x * vb.y) - (va.y * vb.x)))
                self.dirty = True        
            elif((type(va) == Unit_Vector) and (type(vb) == Unit_Vector)):
                ## Cross product of two unit vectors is a unit vector. 
                ident.set_next(Unit_Vector(x = (va.y * vb.z) - (va.z * vb.y),
                                      y = (va.z * vb.x) - (va.x * vb.z),
                                      z = (va.x * vb.y) - (va.y * vb.x),
                                      w = 0.0,
                                      parent_frame = va.parent_frame))
                va.parent_frame.set_eq(vb.parent_frame)
                self.dirty = True

            elif all([(type(v) in [Vector, Unit_Vector]) for v in [va,vb]]):
                ident.set_next(Vector(x = (va.y * vb.z) - (va.z * vb.y),
                                      y = (va.z * vb.x) - (va.x * vb.z),
                                      z = (va.x * vb.y) - (va.y * vb.x),
                                      w = 0.0,
                                      parent_frame = va.parent_frame))
                va.parent_frame.set_eq(vb.parent_frame) 
                self.dirty = True
                
            elif((type(va) in [Vector, Unit_Vector]) and (type(vb) == Mat3x1)):
                ident.set_next(Vector(x = (va.y * vb.z) - (va.z * vb.y),
                                      y = (va.z * vb.x) - (va.x * vb.z),
                                      z = (va.x * vb.y) - (va.y * vb.x),
                                      w = 0.0,
                                      parent_frame = va.parent_frame))
                self.dirty = True

            elif((type(vb) in [Vector, Unit_Vector]) and (type(va) == Mat3x1)):
                ident.set_next(Vector(x = (va.y * vb.z) - (va.z * vb.y),
                                      y = (va.z * vb.x) - (va.x * vb.z),
                                      z = (va.x * vb.y) - (va.y * vb.x),
                                      w = 0.0,
                                      parent_frame = vb.parent_frame))
                self.dirty = True        
            else:
                print("no cross")
                print(va)
                print(vb)
                raise NotImplementedError

        elif (ta['tensor_type'] == TType.UNK) or (tb['tensor_type'] == TType.UNK):
            pass
        else:
            raise Exception("Can't take the cross product of two non-vector types") 

    def run_norm(self, ident : 'ID[Norm]', val : 'Norm[ID]') -> None:
        """
        Normalizes the length of a vector and produces a unit-vector, if
        a unit vector is passed in, it just returns it. 
        """

        t = val.exp.final

        for i in [t]:
            self.tt_alg._run(i, i.val)

        v = t.val

        if (t['tensor_type'] == TType.VECTOR):

            inv_mag = Pow(Mag(v), Constant(-1.0))

            if type(v) == Mat3x1:
                ident.set_next(Mat3x1(x = v.x * inv_mag,
                                      y = v.y * inv_mag,
                                      z = v.z * inv_mag))
                self.dirty = True

            elif type(v) == Mat4x1:
                ident.set_next(Mat4x1(x = v.x * inv_mag,
                                      y = v.y * inv_mag,
                                      z = v.z * inv_mag,
                                      w = v.w * inv_mag))
                self.dirty = True

            elif type(v) in [Vector, Unit_Vector]: 

                ident.set_next(Unit_Vector(x = v.x * inv_mag,
                                           y = v.y * inv_mag,
                                           z = v.z * inv_mag,
                                           w = 0.0,
                                           parent_frame = v.parent_frame))
                self.dirty = True

            else:
                print("no vector norm")
                print(v)

                raise NotImplementedError

        elif (t['tensor_type'] == TType.UNK):
            pass 
        else:
            raise Exception("Can't take the norm of non-vector types") 

    
    def run_dist(self, ident : 'ID[Dist]', val : 'Dist[ID]') -> None:
        """
        Finds the distance between two terms, in the case of scalars
        and points, it's mag(a - b).
        """

        ta = val.exp_a.final
        tb = val.exp_b.final

        for i in [ta,tb]:
            self.tt_alg._run(i, i.val)

        va = ta.val
        vb = tb.val 
        if(ta['tensor_type'] == TType.SCALAR and tb['tensor_type'] == TType.SCALAR):
            ident.set_next(Mag(Add(ta,Negate(tb))))
            self.dirty = True

        elif(ta['tensor_type'] == TType.VECTOR and tb['tensor_type'] == TType.VECTOR):

            if((type(va) == Mat3x1) and (type(vb) == Mat3x1)):
                ident.set_next(Mag(Mat3x1(x = va.x - vb.x,
                                          y = va.y - vb.y,
                                          z = va.z - vb.z)))
                self.dirty = True        
            elif((type(va) == Mat4x1) and (type(vb) == Mat4x1)):
                ident.set_next(Mag(Mat3x1(x = va.x - vb.x,
                                          y = va.y - vb.y,
                                          z = va.z - vb.z,
                                          w = va.w - vb.w)))
                self.dirty = True        
            elif all([(type(v) in {Point,Anchor}) for v in [va,vb]]):
                ident.set_next(Mag(Mat3x1(x = va.x - vb.x,
                                          y = va.y - vb.y,
                                          z = va.z - vb.z)))
                self.dirty = True        
            elif all([(type(v) in [Vector,Unit_Vector]) for v in [va,vb]]):
                ident.set_next(Mag(Mat3x1(x = va.x - vb.x,
                                          y = va.y - vb.y,
                                          z = va.z - vb.z)))
                self.dirty = True
            else:
                print("no dist")
                print(va)
                print(vb)
                raise NotImplementedError

        elif (ta['tensor_type'] == TType.UNK) or (tb['tensor_type'] == TType.UNK):
            pass 
        else:
            raise Exception("Can't take the cross product of non-vector types") 

    def run_inframe(self, ident : 'ID[InFrame]', val : 'InFrame[ID]') -> None:
        """
        Takes one of the object types, and re-renders it in the given
        reference frame.

        works for [Point, Vector, Line, Anchor]
        """

        to = val.obj.final
        tt = val.target_frame.final

        for i in [to,tt]: 
            self.tt_alg._run(i, i.val)

        vo = to.val
        vt = tt.val

        if type(vo) in [Point, Vector, Unit_Vector, Line, Anchor]:

            pf = vo.parent_frame
            trans = pf @ tt.inverse

            if type(vo) in [Point, Vector, Unit_Vector]:
                ident.set_next(trans @ vo)
                self.dirty=True
            elif type(vo) == Line:
                ident.set_next(Line(direction= trans @ vo.direction,
                                    moment = trans @ vo.moment,
                                    parent_frame = tt))
                self.dirty=True
            elif type(vo) == Anchor:
                pt = Point(x=vo.x, y =vo.y, z=vo.z, w=vo.w,
                           parent_frame=vo.parent_frame)
                
                npt = trans @ pt
                ident.set_next(Anchor(x=npt.x,
                                      y=npt.y,
                                      z=npt.z,
                                      w=npt.w,
                                      exists = vo.exists,
                                      direction= trans @ vo.direction,
                                      handle = trans @ vo.handle,
                                      parent_frame=tt))
                self.dirty=True
            else:
                print("no in_frame")
                print(vo)
                print(vt)
                raise NotImplementedError
        else:
            # TODO :: Make sure we can check termination or something for this
            #        eventually. Maybe Anchors, and lines should be promoted
            #        an TType.OBJECT or something?
            pass


    def run_crossmat(self, ident : 'ID[CrossMat]', val : 'CrossMat[ID]') -> None:
        """
        Takes a vector and returns the cross product matrix.

        Nothing super fancy here. 
        """


        t = val.exp.final

        for i in [t]:
            self.tt_alg._run(i, i.val)

        v = t.val

        if (t['tensor_type'] == TType.VECTOR):
            ident.set_next(Mat3x3(c1 = Mat3x1(0.0, v.z, Negate(v.y)),
                                  c2 = Mat3x1(Negate(v.z), 0.0, v.x),
                                  c3 = Mat3x1(v.y, Negate(v.x), 0)))
            self.dirty = True
        elif (t['tensor_type'] == TType.UNK):
            pass 
        else:
            raise Exception("Can't take the crossmat of non-vector type") 

    def run_matmul(self, ident : 'ID[MatMul]', val : 'MatMul[ID]') -> None:
        """
        Matrix multiplication that that works for matrices, vectors,
        frames, and points.

        TODO :: Finish implementing me 
        """


        ta = val.exp_a.final
        tb = val.exp_b.final

        for i in [ta,tb]:
            self.tt_alg._run(i, i.val)

        va = ta.val
        vb = tb.val 

        if(ta['tensor_type'] == TType.MATRIX and tb['tensor_type'] == TType.VECTOR):
            # {Mat3x3, S0Mat3x3} @ {Mat3x1, Vector, Unit_Vector}
            if (type(va) in [Mat3x3, SOMat3x3]):

                if(type(vb) == Mat3x1):
                    ident.set_next(Mat3x1(x = Dot(ta.r1, vb),
                                          y = Dot(ta.r2, vb),
                                          z = Dot(ta.r3, vb)))
                    self.dirty = True
                elif(type(vb) in [Vector,Unit_Vector]):
                    ident.set_next(Vector(x = Dot(ta.r1, vb),
                                          y = Dot(ta.r2, vb),
                                          z = Dot(ta.r3, vb),
                                          w = 0.0,
                                          parent_frame = vb.parent_frame))
                    self.dirty = True
                # elif(type(vb) == Point):
                #     pv = Mat3x1(vb.x, vb.y, vb.z) 
                #     ident.set_next(Point(x = Dot(ta.r1, pv),
                #                          y = Dot(ta.r2, pv),
                #                          z = Dot(ta.r3, pv),
                #                          w = 1.0,
                #                          parent_frame = vb.parent_frame))
                else:
                    print("no matrix vector mul")
                    print(ta.name)
                    print(tb.name)
                    print(va)
                    print(vb)
                    raise NotImplementedError
            elif (type(va) in [Mat4x4, SEMat4x4]): 
                if(type(vb) == Mat4x1):
                    ident.set_next(Mat4x1(x = Dot(ta.r1, vb),
                                          y = Dot(ta.r2, vb),
                                          z = Dot(ta.r3, vb),
                                          w = Dot(ta.r4, vb)))
                    self.dirty = True
                elif(type(vb) in [Vector,Unit_Vector]):
                    ident.set_next(Vector(x = Dot(ta.r1, vb),
                                          y = Dot(ta.r2, vb),
                                          z = Dot(ta.r3, vb),
                                          w = 0.0,
                                          parent_frame = vb.parent_frame))
                    self.dirty = True
                elif(type(vb) in [Point]):
                    ident.set_next(Point(x = Dot(ta.r1, vb),
                                          y = Dot(ta.r2, vb),
                                          z = Dot(ta.r3, vb),
                                          w = 1.0,
                                          parent_frame = vb.parent_frame))
                    self.dirty = True
                else:
                    print("no matrix vector mul")
                    print(va)
                    print(vb)
                    raise NotImplementedError
            elif (type(va) in [Frame, FrameTransform, Body, Hinge, Slide]):

                def_frame = Frame(c1 = Mat4x1(1.0,0.0,0.0,0.0),
                                  c2 = Mat4x1(0.0,1.0,0.0,0.0),
                                  c3 = Mat4x1(0.0,0.0,1.0,0.0),
                                  c4 = Mat4x1(0.0,0.0,0.0,1.0))

                parent_frame = None
                target_frame = None

                if(type(va) in {Frame, Body}):
                    parent_frame = ta
                    target_frame = ta.parent_context.insert(def_frame)
                else:
                    parent_frame = va.parent_frame
                    target_frame = va.target_frame

                if(type(vb) in {Mat4x1}):
                    ident.set_next(Mat4x1(x = Dot(ta.r1, vb),
                                          y = Dot(ta.r2, vb),
                                          z = Dot(ta.r3, vb),
                                          w = Dot(ta.r3, vb)))
                    self.dirty = True
                elif(type(vb) in {Point}):
                    pv = Mat4x1(vb.x, vb.y, vb.z, vb.w) 
                    ident.set_next(Point(x = Dot(ta.r1, pv),
                                         y = Dot(ta.r2, pv),
                                         z = Dot(ta.r3, pv),
                                         w = Dot(ta.r4, pv),
                                         parent_frame = target_frame))
                    parent_frame.set_eq(vb.parent_frame)
                    self.dirty = True
                elif(type(vb) in [Vector,Unit_Vector]):
                    ident.set_next(Vector(x = Dot(ta.r1, vb),
                                          y = Dot(ta.r2, vb),
                                          z = Dot(ta.r3, vb),
                                          w = Dot(ta.r4, vb),
                                          parent_frame = target_frame))
                    parent_frame.set_eq(vb.parent_frame)
                    self.dirty = True
                else:
                    print("no matrix vector mul")
                    print(va)
                    print(vb)
                    raise NotImplementedError
            else:
                print("no matrix vector mul")
                print(va)
                print(vb)
                raise NotImplementedError

        elif(ta['tensor_type'] == TType.MATRIX and tb['tensor_type'] == TType.MATRIX):
            if all([type(v) in [Mat3x3, SOMat3x3] for v in [va,vb]]):

                # We can just use matrix vector multiplication to construct
                # the full matrix multiplication. 
                ident.set_next(Mat3x3(c1=MatMul(va,vb.c1),
                                      c2=MatMul(va,vb.c2),
                                      c3=MatMul(va,vb.c3)))
                self.dirty = True

            elif all([type(v) in [Mat4x4, SEMat4x4] for v in [va,vb]]):

                ident.set_next(Mat4x4(c1=MatMul(va,vb.c1),
                                      c2=MatMul(va,vb.c2),
                                      c3=MatMul(va,vb.c3),
                                      c4=MatMul(va,vb,c4)))
                self.dirty = True

            elif all([(type(v) in [Frame,Body, FrameTransform, Hinge, Slide])
                      for v in [va,vb]]):

                def_frame = Frame(c1 = Mat4x1(1.0,0.0,0.0,0.0),
                                  c2 = Mat4x1(0.0,1.0,0.0,0.0),
                                  c3 = Mat4x1(0.0,0.0,1.0,0.0),
                                  c4 = Mat4x1(0.0,0.0,0.0,1.0))

                parent_frame_a = None
                target_frame_a = None

                if(type(va) in {Frame, Body}):
                    parent_frame_a = ta
                    target_frame_a = ta.parent_context.insert(def_frame)
                else:
                    parent_frame_a = va.parent_frame
                    target_frame_a = va.target_frame

                parent_frame_b = None
                target_frame_b = None

                if(type(vb) in {Frame, Body}):
                    parent_frame_b = tb
                    target_frame_b = tb.parent_context.insert(def_frame)
                else:
                    parent_frame_b = vb.parent_frame
                    target_frame_b = vb.target_frame

                ident.set_next(FrameTransform(c1= MatMul(va, vb.c1),
                                              c2= MatMul(va, vb.c2),
                                              c3= MatMul(va, vb.c3),
                                              c4= MatMul(va, vb.c4),
                                              parent_frame = parent_frame_a,
                                              target_frame = target_frame_b))
                target_frame_a.set_eq(parent_frame_b)
            else:
                print("matmul w/ MATRIX MATRIX")
                print(va)
                print(vb)
                print()
                raise NotImplementedError

        elif (ta['tensor_type'] == TType.UNK) or (tb['tensor_type'] == TType.UNK):
            pass 

        else:
            raise Exception("Can't multiply non-matrix types") 
