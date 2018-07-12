
from mechsynth.term import *
from mechsynth.symbolic.value import *
from mechsynth.symbolic.geom import *
from mechsynth.symbolic.object import *
from mechsynth.symbolic.relation import *
from mechsynth.symbolic.model import *
# from mechsynth.symbolic.model_ops import *
from mechsynth.algebra.small_step import *
from pytest import * # type: ignore
from graphviz import Digraph
from dataclasses import *
from enum import Enum, auto
import itertools
import string
import subprocess
import html


"""
"""

class Assocs(Enum):
    """
    Flags for each of the associative operators we are using.

    ALL and ANY are set based because they're idempotent.
    SUM and PROD require keeping track of constant multiple and power of each term. 
    """
    SUM = auto()
    PROD = auto()
    ALL = auto()
    ANY = auto()

SUM = Assocs.SUM
PROD = Assocs.PROD
ALL = Assocs.ALL
ANY = Assocs.ANY

@model_algebra
class AssocAlg():
    """
    This takes terms like "And", "Or", "Add", and "Mult" and captures
    their associative closures (respectively "All", "Any", "Sum", and "Prod")

    Largely useful for removing duplicates and the like from constraints,
    assumptions, etc...
    """

    def _init_algebra(self, ctxt):
        ctxt.purge_keys({'all', 'any', 'sum', 'prod'})

    def _init_pass(self, ctxt):
        pass 

    def _end_pass(self, ctxt):
        return False

    def _end_algebra(self, ctxt):
        return None

    def _run(self, ident, val):
        if (type(val) == Add):
            # Get the left and right sum dictionaries 
            l_d = AssocAlg.get(SUM, val.exp_a) 
            r_d = AssocAlg.get(SUM, val.exp_b)
            # Iterate over all the elements and keep a running total of
            # how many times they have been encountered. 
            o_d = dict()
            for k in frozenset(l_d.keys()).union(r_d.keys()):
                o_d[k] = l_d.setdefault(k,0) + r_d.setdefault(k,0)
            ident[SUM] = o_d
        elif (type(val) == Mul): 
            # Get the left and right sum dictionaries 
            l_d = AssocAlg.get(PROD, val.exp_a) 
            r_d = AssocAlg.get(PROD, val.exp_b)
            # Iterate over all the elements and keep a running total of
            # how many times they have been encountered. 
            o_d = dict()
            for k in frozenset(l_d.keys()).union(r_d.keys()):
                o_d[k] = l_d.setdefault(k,0) + r_d.setdefault(k,0)
            ident[PROD] = o_d
        elif (type(val) == And):
            ident[ALL] = AssocAlg.get(ALL, val.exp_a).union(
                AssocAlg.get(ALL, val.exp_b))
        elif (type(val) == Or):
            ident[ANY] = AssocAlg.get(ANY, val.exp_a).union(
                AssocAlg.get(ANY, val.exp_b))
        else:
            d = dict()
            d[ident] = 1
            ident[SUM] = d
            ident[PROD] = d
            ident[ALL] = frozenset([ident])
            ident[ALL] = frozenset([ident])
            pass 

    @classmethod
    def get(self, tag, ident):
        alg= AssocAlg()

        if not tag in ident:
            alg._run(ident, ident.val)
        if tag in ident:
            # This is an object of the right type, 
            return ident[tag]
        else:
            # Otherwise it's the vacuous result.
            if tag in {SUM, PROD}:
                d = dict()
                d[ident] = 1
                return d
            else: 
                return frozenset([ident])

def par(s):
    """
    Adds parens if there aren't already parens around and expression. 
    """
    if (s[0] in {'(','{','['}) and (s[-1] in {')','}',']'}):
        return s
    elif len(s) < 8:
        return s
    else:
        return f'({s})' 

def no_par(s):
    """
    Adds parens if there aren't already parens around and expression. 
    """
    if (s[0] in {'('}) and (s[-1] in {')'}):
        return s[1:-1]
    else:
        return s

@dataclass
class ExpData():
    uuid_prefix : str
    relevant    : bool
    min_name    : str
    name_set    : Set[str]
    fn_expr     : str
    short_expr  : str # generally less than a few chars.
    med_expr    : str
    long_expr   : str
    full_expr   : str
    field_lines : Dict[ID,Set[str]]

@model_algebra
class PrintAlg():
    """
    This algebra prints out more readable versions of expressions, and
    generally collates useful metadata together.
    """

    
    def _init_algebra(self, ctxt):
        ctxt.purge_keys('rel_names')
        ctxt.purge_keys('exp_data')

    def _init_pass(self, ctxt):
        pass

    def _end_pass(self, ctxt):
        return False

    def _end_algebra(self, ctxt):
        return None

    @classmethod
    def get(cls, ident):
        alg = PrintAlg()
        if not ('exp_data' in ident):
            alg._run(ident, ident.val)
        return ident['exp_data']

    def _run(self, ident, val):
        ident = id_to_mval(ident)

        name_set = ident.other_names

        e_val = tmap(PrintAlg.get, val)

        uuid_prefix = ident.uuid.hex[:4]

        min_name = None
        if len(ident.other_names) > 0: 
            for name in name_set:
                if min_name == None or (len(min_name) > len(name)):
                    min_name = name
        else:
            min_name = "&" + uuid_prefix


        fields = dict()
        field_lines = dict()

        for f in val.__attrs__:
            # Collect the field values into a dictionary
            fields[self.short_field(f)] = getattr(e_val,f)

            # Collect individual arrows out of this term. If multiple
            # sub-terms point to the same element, consolidate
            # all the terms into a single value. 
            if val.__attrs__[f] == TermType:

                target = getattr(val,f).final
                if target in field_lines:
                    field_lines[target].append(self.short_field(f))
                else:
                    field_lines[target] = list([self.short_field(f)])

        def default_names(ident, e_val, fields):

            class_name =  e_val.__class__.__name__

            def map_fields(fl,f):
                out = list()
                for t in fl:
                    if t != '' and len(fl) > 2:
                        out.append(t + "= " + f(fl[t]))
                    else: 
                        out.append(f(fl[t]))
                return ', '.join(out)

            def shorter(x):
                if len(x.short_expr) < 8:
                    return x.short_expr
                else:
                    return x.min_name

            shr = f'{class_name}(' + map_fields(fields, shorter) + ')'
            med = f'{class_name}(' + map_fields(fields, lambda x: x.short_expr) +')'
            lng = f'{class_name}(' + map_fields(fields, lambda x: x.long_expr)+')'
            ful = f'{class_name}(' + map_fields(fields, lambda x: x.full_expr)+')'

            return (shr, med, lng, ful)

        fun = None

        if hasattr(self, self.language[type(val)]):
            fun = getattr(self, self.language[type(val)])
        else:
            fun = default_names

        def field_names(fields):
            out = list()
            for t in fields.keys():
                    out.append(t)
            return ', '.join(out)

        class_name =  e_val.__class__.__name__
        fn_e = f'{class_name}({field_names(fields)})'


        s_e, m_e, l_e, f_e = fun(ident, e_val, fields)

        if type(e_val) == Constant:
            fn_e = s_e
            min_name = s_e
            uuid_prefix = s_e

        ident['exp_data'] = ExpData(uuid_prefix,
                                    'rel_names' in ident,
                                    min_name,
                                    name_set,
                                    fn_e, s_e, m_e, l_e, f_e,
                                    field_lines)

    def short_field(self, s):
        if s.startswith("exp_"):
            return s[4:] 
        else:
            return { 'exp' : '',
                     'initial_condition' : 'init',
                     'parent_frame' : 'frame',
                     'target_frame' : 'target'}.get(s, s)
        

    def run_param(self, ident, e_val, fields):

        typestr = None

        if e_val.v_type == ValType.REAL:
            typestr = 'flt'
        elif e_val.v_type == ValType.BOOL:
            typestr = 'bool'
        elif e_val.v_type == ValType.INT:
            typestr = 'int'

        s_n = ident.name
        l_n = f'{ident.name} ::{typestr})'

        return (s_n, s_n, l_n, l_n)

    def run_atinitial(self, ident, val, fields):

        def ini(ex):
            nonlocal val
            return  f'init{par(ex(val.exp))}'

        return self.run_versions(ini,shift=0)
        

    def run_frame(self, ident, e_val, fields):

        if ident.is_relevant:
            n = ident.name
        else:
            n = '&' + ident.uuid.hex[:4]

        n = f'{n}::frm'

        return (n, n, n, n)

    def run_variable(self, ident, val, field):
        if ident.is_relevant:
            n = ident.name
        else:
            n = '&' + ident.uuid.hex[:4]

        n = f'{n}::var'

        return (n, n, n, n)

    def run_control(self, ident, val, field):
        if ident.is_relevant:
            n = ident.name
        else:
            n = '&' + ident.uuid.hex[:4]

        n = f'{n}::cntl'

        return (n, n, n, n)

    def run_constant(self, ident, val, fields):

        s_n = repr(val.const_val)

        return (s_n, s_n, s_n, s_n)

    def run_versions(self, fun, shift=1):

        s_n, m_n, l_n, f_n = (None, None, None, None)

        if shift == 1:
            s_n = fun(lambda x: x.min_name)
            m_n = fun(lambda x: x.short_expr)
            l_n = fun(lambda x: x.med_expr)
            f_n = fun(lambda x: x.full_expr)
        elif shift == 2:
            s_n = fun(lambda x: x.min_name)
            m_n = fun(lambda x: x.min_name)
            l_n = fun(lambda x: x.short_expr)
            f_n = fun(lambda x: x.full_expr)
        elif shift == 3:
            s_n = fun(lambda x: x.min_name)
            m_n = fun(lambda x: x.min_name)
            l_n = fun(lambda x: x.min_name)
            f_n = fun(lambda x: x.full_expr)
        else: 
            s_n = fun(lambda x: x.short_expr)
            m_n = fun(lambda x: x.med_expr)
            l_n = fun(lambda x: x.long_expr)
            f_n = fun(lambda x: x.full_expr)

        return (s_n, m_n, l_n, f_n) 

    def run_getmember(self, ident, val, fields):

        def add_member(ex):
            nonlocal val
            return  f'{par(ex(val.exp))}.{val.member}'

        return self.run_versions(add_member)

    def run_ifthenelse(self, ident, val, fields):

        def ite(ex):
            nonlocal val
            return (f'({par(ex(val.exp_true))} if {par(ex(val.exp_cond))} '
                        + f'else {par(ex(val.exp_false))})')

        return self.run_versions(ite)
    
    def run_negate(self, ident, val, fields):

        def neg(ex):
            nonlocal val
            return  f'-{ex(val.exp)}'

        return self.run_versions(neg,shift=0)

    def run_invert(self, ident, val, fields):

        def inv(ex):
            nonlocal val
            return  f'{ex(val.exp)}^(-1)'

        return self.run_versions(inv,shift=0)

    def run_not(self, ident, val, fields):

        def notf(ex):
            nonlocal val
            return  f'not {ex(val.exp)}'

        return self.run_versions(notf,shift=0)

    def run_eq(self, ident, val, fields):

        def notf(ex):
            nonlocal val
            return  f'({ex(val.exp_a)} == {ex(val.exp_b)})'

        return self.run_versions(notf)

    def run_neq(self, ident, val, fields):

        def notf(ex):
            nonlocal val
            return  f'({ex(val.exp_a)} != {ex(val.exp_b)})'

        return self.run_versions(notf)

    def run_lessthan(self, ident, val, fields):

        def coll_terms(ex):
            nonlocal val
            return  f'({ex(val.exp_a)} < {ex(val.exp_b)})'

        return self.run_versions(coll_terms)

    def run_lessthaneq(self, ident, val, fields):

        def coll_terms(ex):
            nonlocal val
            return  f'({ex(val.exp_a)} <= {ex(val.exp_b)})'

        return self.run_versions(coll_terms)


    def run_greaterthan(self, ident, val, fields):

        def coll_terms(ex):
            nonlocal val
            return  f'({ex(val.exp_a)} > {ex(val.exp_b)})'

        return self.run_versions(coll_terms)

    def run_greaterthaneq(self, ident, val, fields):

        def coll_terms(ex):
            nonlocal val
            return  f'({ex(val.exp_a)} >= {ex(val.exp_b)})'

        return self.run_versions(coll_terms)

    def run_implies(self, ident, val, fields):

        def coll_terms(ex):
            nonlocal val
            return  f'({ex(val.exp_a)} -> {ex(val.exp_b)})'

        return self.run_versions(coll_terms)

    def run_matmul(self, ident, val, fields):

        def coll_terms(ex):
            nonlocal val
            return  f'({ex(val.exp_a)} @ {ex(val.exp_b)})'

        return self.run_versions(coll_terms)

    def run_pow(self, ident, val, fields):

        def coll_terms(ex):
            nonlocal val
            return  f'({ex(val.exp_a)} ** {ex(val.exp_b)})'

        return self.run_versions(coll_terms)

    def run_add(self, ident, val, fields):


        def coll_terms(ex):
            nonlocal val
            return  f'({ex(val.exp_a)} + {ex(val.exp_b)})'

        return self.run_versions(coll_terms)

    def run_sub(self, ident, val, fields):

        def coll_terms(ex):
            nonlocal val
            return  f'({ex(val.exp_a)} - {ex(val.exp_b)})'

        return self.run_versions(coll_terms)

    def run_mul(self, ident, val, fields):

        def coll_terms(ex):
            nonlocal val
            return  f'({par(ex(val.exp_a))} * {par(ex(val.exp_b))})'

        # terms = AssocAlg.get(PROD, ident)

        # def coll_terms(ex):
        #     nonlocal terms
        #     l = list()
        #     for k in terms:
        #         if terms[k] != 1: 
        #             l.append(f'{par(ex(PrintAlg.get(k)))} ** {terms[k]}')
        #         else:
        #             l.append(ex(PrintAlg.get(k)))

        #     return ' * '.join(l)

        return self.run_versions(coll_terms)

    def run_div(self, ident, val, fields):

        def coll_terms(ex):
            nonlocal val
            return  f'({ex(val.exp_a)} - {ex(val.exp_b)})'

        return self.run_versions(coll_terms)

    def run_and(self, ident, val, fields):

        terms = AssocAlg.get(ALL, ident)

        def coll_terms(ex):
            nonlocal terms
            l = list()
            for k in terms:
                l.append(f'{par(ex(PrintAlg.get(k)))}')
            if len(l) > 1:
                return f'({"&&".join(l)})'
            else:
                return '&&'.join(l)

        return self.run_versions(coll_terms)

    def run_or(self, ident, val, fields):

        terms = AssocAlg.get(ALL, ident)

        def coll_terms(ex):
            nonlocal terms
            l = list()
            for k in terms:
                l.append(f'{par(ex(PrintAlg.get(k)))}')
            if len(l) > 1:
                return f'({"||".join(l)})'
            else:
                return '||'.join(l)

        return self.run_versions(coll_terms)

    def run_mat3x1(self, ident, val, fields):

        def m31(ex):
            pe = lambda x: par(ex(x))
            nonlocal val
            return f'[{pe(val.x)}, {pe(val.y)}, {pe(val.z)}]'

        return self.run_versions(m31)

    def run_mat4x1(self, ident, val, fields):

        def m41(ex):
            pe = lambda x: no_par(ex(x))
            nonlocal val
            return f'[{pe(val.x)}, {pe(val.y)}, {pe(val.z)}, {pe(val.w)}]'

        return self.run_versions(m41)

    def run_point(self, ident, val, fields):

        def m41(ex):
            pe = lambda x: no_par(ex(x))
            nonlocal val
            return f'Pt({pe(val.x)}, {pe(val.y)}, {pe(val.z)}, {pe(val.w)})'

        return self.run_versions(m41)

    def run_vector(self, ident, val, fields):

        def m41(ex):
            pe = lambda x: no_par(ex(x))
            nonlocal val
            return f'Vec({pe(val.x)}, {pe(val.y)}, {pe(val.z)}, {pe(val.w)})'

        return self.run_versions(m41)

    def run_unit_vector(self, ident, val, fields):

        def m41(ex):
            pe = lambda x: no_par(ex(x))
            nonlocal val
            return f'UnitVec({pe(val.x)}, {pe(val.y)}, {pe(val.z)}, {pe(val.w)})'

        return self.run_versions(m41)


def pretty_graphviz(model,
                    expand_nexts = False):
    """
    A significantly prettier version of the graphviz diagram that uses
    the parse_alg's names and stuff.

    params:
      expand_nexts := If true shows every full chain, otherwise it only
                      shows the final element.
    """
    # TODO :: - Create special nodes for key terms in the model,
    #        - sort things into equality sets
    #          - split equality set into chains
    #          - if expand nexts, make each chain a group w/ the chain of
    #            terms, otherwise make a single node with the final term.
    #          - Add connections between equal elements.
    #        - add links between each node and its members
    #        - add links for each term in the key-terms

    model.run_algebra(TTypeAlg()) 
    model.run_algebra(PrintAlg()) 

    graph = Digraph(model.name,
                    engine='osage',
                    graph_attr=dict(label=model.name,
                                    concentrate='true',
                                    clusterrank='local'))

    g_node_style = dict(shape='box',
                        style='rounded',)

    eq_edge_style = dict(constraint='false',
                         color='grey73:grey73',
                         dir='none')
    next_edge_style = dict(penwidth='2', decorate='true')
    member_edge_style = dict() 


    def tensor_style(sty, ident):
        nonlocal model

        sty = sty.copy()
        fill_color = None
        font_color = None
        color = list()
        pen_width = None

        if ident['tensor_type'] == TType.SCALAR:
            fill_color = 'darkolivegreen1'
        elif ident['tensor_type'] == TType.VECTOR:
            fill_color = 'lightblue1'
        elif ident['tensor_type'] == TType.MATRIX:
            fill_color = 'plum1'

        if type(ident.val) in {Param}:
            fill_color = 'dodgerblue4'
            font_color = 'white'
        elif type(ident.val) in {Variable}:
            fill_color = 'darkorchid4'
            font_color = 'white'
        elif type(ident.val) in {Control}:
            fill_color = 'deeppink4'
            font_color = 'white'
        elif type(ident.val) in {Constant}:
            fill_color = 'khaki'

        def group_color(s, c):
            nonlocal ident
            nonlocal color
            nonlocal pen_width
            if ident in s:
                color.append(c)
                pen_width = '2'
    
        group_color(model.objects, 'deeppink')
        group_color(model.ref_frames, 'crimson')
        group_color(model.parameters, 'deepskyblue')
        group_color(model.controls, 'firebrick1')
        group_color(model.variables, 'darkorchid1')
        group_color(model.assertions, 'darkgreen')
        group_color(model.assumptions, 'firebrick4')
        group_color(model.constraints, 'darkgreen')
        group_color(model.guarantees, 'dodgerblue3')
        group_color(model.cost_terms, 'darkorchid4')


        if fill_color != None:
            sty['fillcolor'] = fill_color
            if not 'filled' in sty['style']:
                sty['style'] = sty['style'] + ", filled"

        if font_color != None:
            sty['fontcolor'] = font_color

        if pen_width != None:
            sty['penwidth'] = pen_width

        if len(color) >= 1:
            sty['peripheries'] = str(len(color))
            sty['color'] = ':'.join(color)

        return sty


    # Gather up equality classes 
    equality_sets = set()
    for term in model.topolist():
        term = id_to_mval(term)
        if 'eq_set' in term:
            equality_sets.add(frozenset(term['eq_set']))
        else:
            equality_sets.add(frozenset([term]))

    # split equality class into a set of chains based on their 
    chains = list()
    for eq_class in equality_sets:
        chain_set = dict()
        for term in eq_class:
            if term.final in chains:
                chain_set[term.final].add(term)
            else:
                chain_set[term.final] = set([term])
        chains.append(chain_set) 

    # The graphs we need to add member edges to
    nodes = dict()
    member_edge_queue = set()
    consts = set()

    def add_node(graph, ident, *args, **kwargs):
        nonlocal nodes
        nodes[ident] = ident.uniq_name
        graph.node(ident.uniq_name, *args, **kwargs)

    def add_edge(frm, to, *args, **kwargs): 
        nonlocal nodes
        nonlocal graph

        f = nodes[frm]
        t = nodes[to]
        # if nodes[to][1] < nodes[frm][1]:
        #    kwargs['constraint'] = 'false'
        graph.edge(f,t, *args, **kwargs) 

    for chain_set in chains:
        for final, elems in chain_set.items():

            member_edge_queue.add(final)

            members = set()

            members.add(final)

            if expand_nexts:
                members.update(elems)
                temp = set()
                for elem in  members:
                    if ('next' in elem) and (elem['next'] != None):
                        temp.add(elem['next'])
                members.update(temp)


            def add_nodes(g, members):

                nonlocal g_node_style
                nonlocal consts

                for node in members: 
                    name_set = None

                    pt = PrintAlg.get(node)

                    html_used = len(node.other_names) > 0

                    if len(node.other_names) > 0:
                        name_set = node.other_names
                    else:
                        mn = pt.min_name
                        if mn[0] == '&':
                            mn = mn[1:]
                        name_set = set([mn])

                    lval = no_par(pt.med_expr)

                    if len(lval) >= 80:
                        lval = no_par(pt.short_expr)
                                
                    # if len(lval) >= 80:
                    #     lval = no_par(pt.uuid_prefix)

                    lname = None
                    if not html_used: 
                        lname = ''.join(name_set)
                    else:
                        lname = '<BR/>'.join(map(html.escape,name_set))
                        lname = f'<B>{lname}</B><BR/><BR/>'

                    if all(c in string.hexdigits for c in lname) and not html_used:
                        lname = f'*{lname}'

                    sty = g_node_style.copy()

                    if node != node.final:
                        sty['color'] = 'grey'
                        sty['fontcolor'] = 'grey35'
                                        
                    label = None

                    if type(node.val) == Constant:
                        if html_used:
                            label = f'{lname} := {lval}'
                        else:
                            label = f'{node.val.const_val}'
                        consts.add(node)
                    elif type(node.val) == Param:
                        label = f'{lname} :: Parameter'

                    elif type(node.val) == Variable:
                        label = f'{lname} :: Variable'
                    elif type(node.val) == Control:
                        label = f'{lname} :: Control'
                    else:
                        lval = f' := {lval}'
                        if html_used:
                            lval = html.escape(lval)
                        label = lname + lval

                    if html_used:
                        label = f'<{label}>'

        
                    add_node(g, node,
                             label=label,
                             **tensor_style(sty,node))

            if expand_nexts and len(members) > 1:
                with graph.subgraph(name=f'cluster_{final.uniq_name}') as sg:
                    pt = PrintAlg.get(final)
                    name_set = None
                    if len(final.other_names) > 0:
                        name_set = final.other_names
                    else:
                        mn = pt.min_name
                        if mn[0] == '&':
                            mn = mn[1:]
                        name_set = set([mn])
                    lname = '<BR/>'.join(name_set)
                    sg.attr(label=f'<{lname}>', style='rounded', color='grey75')
                    add_nodes(sg, members)
            else:
                add_nodes(graph,members)

    
    for chain_set in chains:
        if expand_nexts:
            for final, elems in chain_set.items():
            # then the edges
                for elem in elems:
                    if ('next' in elem) and (elem['next'] != None): 
                        sty = next_edge_style.copy()
                        sty['color']= elem['next_col']

                        add_edge(elem,elem['next'], **sty) 

        if len(chain_set) > 1:
            for na, nb in itertools.combinations(chain_set.keys(), 2):
                add_edge(na, nb, **eq_edge_style)


    # Add the edges for the terms we use. 
    for ident in member_edge_queue:
        fields = PrintAlg.get(ident).field_lines

        for f in fields.keys():
            label = ', '.join(fields[f])
            sty = member_edge_style.copy()
            if len(label) > 1:
                sty['label'] = label
            if type(f.final.val) != Constant: 
                add_edge(ident,f.final, **sty)

            #if expand_nexts:
            #    graph.edge(ident.uniq_name, f.uniq_name, style='invis')


    # Add the top level nodes
    model_node_style = dict(shape='box',
                            style='filled',
                            fillcolor='gold')

    model_elem_style = dict(shape='box',
                            style='filled',
                            fillcolor='goldenrod1')

    model_edge_style = dict(dir='none')
    model_elem_edge_style = dict(dir='none', color='grey')

    graph.node('Model', **model_node_style)

    def make_terms(mset, name, label):

        nonlocal graph
        nonlocal model_elem_style
        nonlocal model_elem_edge_style
        nonlocal expand_nexts

        if len(mset) > 0:
            graph.node(name,label=label,**model_elem_style)

            graph.edge('Model', name, **model_edge_style)

            for f in mset:
                if expand_nexts:
                    graph.edge(name, f.uniq_name, **model_elem_edge_style)
                else:
                    graph.edge(name, f.final.uniq_name, **model_elem_edge_style)


    def make_end_terms(mset, name, label):

        nonlocal graph
        nonlocal model_elem_style
        nonlocal model_elem_edge_style

        if len(mset) > 0:
            graph.node(name,label=label,**model_elem_style)

            for f in mset:
                graph.edge(f.final.uniq_name, name, **model_elem_edge_style)
    
    make_terms(model.ref_frames, 'ref_frames', 'Rigid Frames')
    make_terms(model.objects, 'objects', 'Objects')
    make_end_terms(model.parameters, 'params', 'Parameters')
    make_end_terms(model.controls, 'controls', 'Controls')
    make_end_terms(model.variables, 'variables', 'Variables')
    make_terms(model.assertions, 'assertions', 'Assertions')
    make_terms(model.assumptions, 'assumptions', 'Assumptions')
    make_terms(model.constraints, 'constraints', 'Constraints')
    make_terms(model.guarantees, 'guarantees', 'Guarantees')
    make_terms(model.cost_terms, 'cost_terms', 'Costs')
    make_terms(consts, 'constants', 'Constants')
    
    return graph

                    
def unflatten(s):
    out = subprocess.run(["unflatten","-f","-l 20"],
                         input=s,
                         stdout=subprocess.PIPE,
                         universal_newlines=True)
    out.check_returncode()
    return out.stdout

def print_graph(name, model, *args, **kwargs):
    pathlib.Path('__debug/').mkdir(parents=True, exist_ok=True)
    f = open(f'__debug/{name}.dot',"w")
    f.write(pretty_graphviz(model,*args, **kwargs).source)
    f.close()

def print_graphs(name, model, *args, **kwargs):
    print(f'printing {name}')
    #print_graph(name, model, *args, **kwargs)
    print_graph(f'{name}', model, *args, expand_nexts=True, **kwargs)

@model_algebra
class PruneAlg():
    """
    This algebra prunes a model into a smaller logically equivalent one, by
    taking only the final terms that are relevant to the operation.

    also this does some basic checks 

    Steps:
      - Generate Terms
      - Generate Flat Versions of each thing, (creating new vars, params, and
        controls as needed)
      - Gather sets of major objects
      - Use AssocAlg to minimize the constraint/etc.. term reps
    """

    new_ctxt : Model
    ctxt_tag : str
    dirty : bool = False

    keep_relevant : bool

    def __init__(self, alg_name, keep_relevant = True):
        
        self.new_ctxt = Model(alg_name)
        self.ctxt_tag = f'in_{alg_name}'
        self.keep_relevant = keep_relevant
        # add the assumptions, constraints, and costs 

    def _init_algebra(self, ctxt):
        ctxt.purge_keys(self.ctxt_tag)
        # get all assumption, guarantee and cost terms after running the
        # assoc algebra and stuff.

        def copy_field(f_name, get_field, alg, eqs=True):

            nonlocal ctxt
            nonlocal self

            for term in get_field(ctxt):
                for t in AssocAlg.get(alg, term.final):

                    # We care about this term, move it over
                    new_term = self.flat(t, False)

                    # Copy over names, so that we can keep track of how things
                    # derived (at least a little)
                    for n in term.other_names:
                        new_term.add_name(n, is_rel=term.is_relevant)

                    # If we create something of the wrong type, or if we learn
                    # that some term reduced to false, throw an error. 
                    if type(new_term.val) == Constant:
                        if new_term.val.const_val == True:
                            pass
                        elif new_term.val.const_val == False:
                            print(f'{f_name} {term} reduced to False')
                        else:
                            raise TypeError(f'{f_name} reduced to: \n'
                                            + f'  term: '
                                            + f'{term}'
                                            + f'\n  type: '
                                            + f'{type(new_term.val.const_val)}'
                                            + f'\n  val : '
                                            + f'{new_term.val.const_val}')

                    # Add the term to the field in the new context
                    get_field(self.new_ctxt).add(new_term)

                    # If there's a number of equivalent terms, do the same
                    # thing for each equivalent term. 
                    # if eqs and ('eq_set' in t) and (t['eq_set'] != None):
                    #     for e in t['eq_set']:
                    #         new_eq = self.flat(e)
                    #         for n in term.other_names:
                    #             new_eq.add_name(n, is_rel=term.is_relevant)
                    #         get_field(self.new_ctxt).add(new_eq)
                    #         # Make sure we keep track of the equality
                    #         # constraint here. 
                    #         new_term.set_eq(new_eq)

        copy_field('assertion', lambda x: x.assertions, ALL)
        copy_field('assumption', lambda x: x.assumptions, ALL)
        copy_field('constraint', lambda x: x.constraints, ALL)
        copy_field('guarantee', lambda x: x.guarantees, ALL)

        for term in ctxt.cost_terms:
            self.new_ctxt.cost_terms.add(self.flat(term))

    def _init_pass(self, ctxt):
        pass

    def _end_pass(self, ctxt):
        return False

    def _end_algebra(self, ctxt):
        # Remove all those extra references in the original context.
        ctxt.purge_keys(self.ctxt_tag)
        return self.new_ctxt

    def _run(self, ident, val):
        """
        Basically if something is relevant, insert its final version into
        the new ctxt.

        Otherwise we do nothing.

        This should only really run once. 
        """

        ident = id_to_mval(ident)


        # Propagate relevant terms forward.
        if self.keep_relevant and len(ident.other_names) > 0:
            nt = self.flat(ident.final)
            for name in ident.other_names: 
                nt.add_name(name, is_rel=True)

        # Promote any equalities we find to constraints.
        # NOTE :: Yeah, this is a bit iffy, esp since we should see a
        #        lot of these terms just collapsing into a single flat
        #        term. Hopefully this will lead to a lot of attempted
        #        duplicate insertions.
        # NOTE :: We also don't flag promotion as dirtying, since ....
        #        well, I can't think of another way to keep it from
        #        infinite looping. 
        # if (ident.final == ident) and ('eq_set' in ident.final):
        #     finals = set()

        #     for item in ident.final['eq_set']:
        #         finals.add(id_to_mval(item.final))

        #     print(ident['eq_set'])

        #     combs = list(itertools.combinations(finals, 2))
        #     for na, nb in combs:
        #         #print(f'promoting eq_set equality:\n\n{na}\n\n{nb}\n\n')
        #     
        #         ma = PrintAlg.get(na).min_name
        #         mb = PrintAlg.get(nb).min_name 

        #         self.new_ctxt.add_constraint(
        #             self.flat(na).equals(self.flat(nb)),
        #             name=f'promoted-eq({ma}, {mb})')
        # else:
        #     pass

        # Preserve variables and params, and keep them pointed to their latest
        # versions. That way, even if we have functionalization happening, we
        # get useful output. 
        if ((ident != ident.final)
            and not (self.ctxt_tag in ident)
            and (type(ident.val) in {Param,Control,Variable})):

            #print('preserving pcv') 
            term = self.flat(ident, False)
            term.set_next(self.flat(ident))

    def flat(self, ident, final = True):
        if final:
            ident = id_to_mval(ident.final)
        else:
            #print(f'inserting self')
            ident = id_to_mval(ident)

        # Make sure we insert the final version of each element. 
        if not (self.ctxt_tag in ident):

            #print(f'adding {ident.uniq_name}')
            val = ident.val

            name = ident.name
            for n in ident.other_names:
                if len(n) <= len(name):
                    name = n

            if type(val) == Param:
                ident[self.ctxt_tag] = self.new_ctxt._param(
                    name = ident.name,
                    type_hint = val.v_type)
            elif type(val) == Control:
                ident[self.ctxt_tag] = self.new_ctxt._control(
                    name = ident.name,
                    initial_condition = self.flat(val.initial_condition))
            elif type(val) == Variable:
                ident[self.ctxt_tag] = self.new_ctxt._variable(
                    name = ident.name,
                    initial_condition = self.flat(val.initial_condition))
            else:
                name = None
                if len(ident.other_names) != 0:
                    name = ident.name
                ident[self.ctxt_tag] = self.new_ctxt.insert(tmap(self.flat,val),
                                                            name=name)

        return ident[self.ctxt_tag]
