from mechsynth.term import *
from mechsynth.symbolic.value import *
from mechsynth.symbolic.geom import *
from mechsynth.symbolic.object import *
from mechsynth.symbolic.relation import *
from mechsynth.symbolic.model import *
from mechsynth.symbolic.model_ops import *
from public import public 

pi = 3.14159 
# TODO :: test_mval_hash_eq

@public
def find_orthogonal():
    """
    We want to find assignments for a thru d such that `dot((x,y), pt) == 0` 
    """

    ang = control("theta")

    assume(ang > -2.0*pi,
           ang < 2.0*pi,
           name='theta-bounds')

    pt = mat3x1(sin(ang), cos(ang), 0.0)

    a = parameter('a',bool)
    b = parameter('b',bool)
    c = parameter('c',bool)
    d = parameter('d',bool)

    x = if_then_else(a,sin(ang),cos(ang)) * if_then_else(b, 1.0, -1.0)
    y = if_then_else(c,sin(ang),cos(ang)) * if_then_else(d, 1.0, -1.0)

    op = mat3x1(x,y, 0.0)

    require(dot(pt,op).equals(0.0),name='is-ortho')

@public
def design_pantograph():
    """
    We assume that a pantograph is made up of a series of 4 rigid bodies
    attached together.

    There are 4 rigid bodies ('A','B','C', and 'D') along with the
    parent frame, as well as 5 hinges between those bodies ('a' thru 'e').

    We want to make sure that the vector from the output to its origin
    is some known function of the vector from the pen to its origin. 

    ```
               a           D       e
           +-------------------+------------------+ out()
           |                   |
           |                   |
           |                   |
           |                   |
         A |                   |C
           |                   |
           |                   |
           |                   |
           |                   |
           |                   |
          b+-------------------+ pen()
           |         B         d
           |
           |
           |
           |
           |
           |
           |
           |
           |
          c+ 
    ```

    We want the tool find locations and angle limitations for hinges that
    allow the pantograph to correctly duplicate things. 

    """

    # We are working in cm.  
    cm = lambda x : const(x)

    # The center of the pen's drawing area 
    input_origin = point("input_origin", fixed=True)
    # the radius of the drawing area
    input_radius = cm(20) 

    # center of the output's drawing area
    output_origin = point("output_origin", fixed=True)
    output_scale = const(3.0)
    output_err   = cm(0.1)
    output_radius = input_radius * output_scale + const(2.0) * output_err

    # Create each of the rigid bodies. 
    arm = dict()

    for n in ['A','B','C','D']:
        arm[n] = rigid_body(f'arm_{n}', exists=True)

    # We create the pen and output as anchors in different frames but we
    # want to work with them as objects in the default frame.

    pen_x = control('pen_x')
    pen_y = control('pen_y') 

    pen = anchor('pen', body=arm['B']).in_frame(default_frame())

    out = anchor('out', body=arm['D']).in_frame(default_frame())


    # We want to ensure that the position of the pen (at all times) is
    # mapped to our controls
    constrain(pen_x.equals(pen.x),
              pen_y.equals(pen.y),
              name='pen-position')

    # and that both the pen and the output are constrained to the z plane.
    require(pen.z.equals(0.0),
            out.z.equals(0.0),
            name='z-plane-req')

    # Then we create the hinges, from a list we manually populate
    hinges = dict()

    hinge_list = [('a',arm['A'],arm['D']),
                  ('b',arm['A'],arm['B']),
                  ('c',arm['A'],default_frame()),
                  ('d',arm['B'],arm['C']),
                  ('e',arm['C'],arm['D'])]

    for h, f_from, f_to in hinge_list:
        name = f'hinge_{h}'
        hinges[h] = hinge(name, f_from, f_to)

    # We are going to assume that the input is always within a given. 
    assume(pen.within(dist=input_radius, of=input_origin),
           name='writing-area-assumption')

    # We don't want the input and output area to overlap, so ask the tool to
    # choose parameters that keep those farther apart. 
    constrain(m_not(input_origin.within(dist= input_radius + output_radius + 2.0,
                                        of= output_origin)),
              name='non-overlapping-writing')

    # And this is our core requirement, namely that output is an accurate
    # reproduction of the input given some scaling factor. 
    require((out.loc - output_origin).within(dist=output_err,
                                             of=output_scale
                                                * (pen.loc - input_origin)),
            name='reproduction-req')

    def point_nearest_origin(line):
        # given how we represent a line (a moment about the origin, and unit
        # vector for direction) we can generate the point nearest the origin
        # with:  cross(dir, moment)/dot(dir,dir)
        return point(line.name + ".origin_pt",
                    line.dir.cross(line.moment) / dot(line.dir, line.dir))

    # Finally, we want to minimize the size of the entire assembly, as
    # defined by sum of the distances between the input origin, and those
    # points on the hings axes closest to the origin. 
    penalize(origin().dist_to(input_origin),
             origin().dist_to(output_origin),
             origin().dist_to(point_nearest_origin(hinges['a'].axis)),
             origin().dist_to(point_nearest_origin(hinges['b'].axis)),
             origin().dist_to(point_nearest_origin(hinges['c'].axis)),
             origin().dist_to(point_nearest_origin(hinges['d'].axis)),
             origin().dist_to(point_nearest_origin(hinges['e'].axis)),
             name='location-dist-penalties')

@public
def single_hinge():


    angle = control('angle', max_bound= 2*pi, min_bound=-2*pi)  
    arm = rigid_body('arm', exists=True)
    pivot = hinge('pivot', default_frame(), arm, angle = angle)
    point = anchor('point', body=arm).in_frame(default_frame())

    require(point.within(dist= 10.0, of=origin()),
            name='dist-req')

test_data_list = [
    ("find_orthogonal", find_orthogonal, {}),
    ("design_pantograph", design_pantograph ,{}),
    ("single_hinge", single_hinge, {}),
    ]


@public
def test_data(flags = None):
    """
    Takes the test data and test ids from the above, and prunes out those
    with 
    """

    filtr = None

    if flags == None:
        filtr = (lambda x: True)
    elif type(flags) == bool:
        filtr = (lambda x: flags)
    elif type(flags) == str:
        filtr = (lambda x: flags in x)
    elif isinstance(flags, Iterable):
        flag_set = frozenset(flags)
        filtr = (lambda x: not flags.isdisjoint(x))
    else:
        raise TypeError(f'{type(flags)} is not a supported type for flags')

    tests = list()
    names = list()

    for (name, builder, flags) in test_data_list:
        if filtr(flags):
            tests.append((name, builder))
            names.append(name)

    return {'argvalues' : tests, 'ids' : names}


