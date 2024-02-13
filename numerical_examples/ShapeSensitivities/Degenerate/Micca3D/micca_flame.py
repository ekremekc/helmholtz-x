from math import pi, radians, cos, sin
import gmsh

import micca_flame_physical_groups as physical


def reflection_matrix():
    return [1,  0,  0,  0,
            0, -1,  0,  0,
            0,  0,  1,  0,
            0,  0,  0,  1]


def rotation_matrix(angle):
    c, s = cos(angle), sin(angle)
    return [c, -s,  0,  0,
            s,  c,  0,  0,
            0,  0,  1,  0,
            0,  0,  0,  1]


def geom_1(file, fltk=False, **kwargs):

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(__name__)

    # geom = gmsh.model.geo

    add_elementary_entities(**kwargs)
    apply_symmetry()
    apply_rotation()
    add_physical_entities()

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)

    gmsh.option.setNumber("Mesh.SaveAll", 0)
    # gmsh.option.setNumber("Mesh.MshFileVersion", 2)
    gmsh.write("{}.msh".format(file))

    if fltk:
        fltk_options()
        gmsh.fltk.run()

    # gmsh.finalize()


def add_elementary_entities(**kwargs):
    """
    inner : in
    middle:
    outer : out
    p : plenum
    b : burner
    pp: injection
    f : flame
    cc: combustion chamber
    """
    
    # The default parameters
    params = {'R_in_p': .14,
              'R_out_p': .21,
              'l_p': .07,
              'h_b': .0165,
              'l_b': .014,
              'h_pp': .00945,
              'l_pp': .006,
              'h_f': .018,
              'l_f': .006,
              'R_in_cc': .15,
              'R_out_cc': .2,
              'l_cc': .2,
              'l_ec': 0.041,
              'lc_1': 1e-1,
              'lc_2': 2e-2
              }

    # Loop for changing default mesh parameters by taken keyword argument(dict)
    for key, value in kwargs.items():
        if key in params.keys():
            params[key] = value

    R_in_p = params['R_in_p']
    R_out_p = params['R_out_p']
    l_p = params['l_p']

    h_b = params['h_b']
    l_b = params['l_b']

    h_pp = params['h_pp']
    l_pp = params['l_pp']

    h_f = params['h_f']
    l_f = params['l_f']

    R_in_cc = params['R_in_cc']
    R_out_cc = params['R_out_cc']
    l_cc = params['l_cc']
    l_ec = params['l_ec']  # end correction

    R_mid = .175  # mid plane

    lc_1 = params['lc_1']
    lc_2 = params['lc_2']
    
    theta = 22.5 / 2
    theta = radians(theta)

    geom = gmsh.model.geo

    # ________________________________________________________________________________
    # PLENUM

    p1 = geom.addPoint(0., 0., - l_pp - l_b - l_p, lc_1)

    p2 = geom.addPoint(R_in_p, 0., - l_pp - l_b - l_p, lc_1)
    p3 = geom.addPoint(R_out_p, 0., - l_pp - l_b - l_p, lc_1)
    p4 = geom.addPoint(R_in_p * cos(theta), R_in_p * sin(theta), - l_pp - l_b - l_p, lc_1)
    p5 = geom.addPoint(R_out_p * cos(theta), R_out_p * sin(theta), - l_pp - l_b - l_p, lc_1)

    l1 = geom.addLine(p2, p3)
    l2 = geom.addCircleArc(p3, p1, p5)
    l3 = geom.addLine(p5, p4)
    l4 = geom.addCircleArc(p4, p1, p2)

    ll1 = geom.addCurveLoop([-l1, -l4, -l3, -l2])
    s1 = geom.addPlaneSurface([ll1])

    p6 = geom.addPoint(0., 0., - l_pp - l_b, lc_1)

    p7 = geom.addPoint(R_in_p, 0., - l_pp - l_b, lc_1)
    p8 = geom.addPoint(R_out_p, 0., - l_pp - l_b, lc_1)
    p9 = geom.addPoint(R_in_p * cos(theta), R_in_p * sin(theta), - l_pp - l_b, lc_1)
    p10 = geom.addPoint(R_out_p * cos(theta), R_out_p * sin(theta), - l_pp - l_b, lc_1)

    l5 = geom.addLine(p2, p7)
    l6 = geom.addLine(p3, p8)
    l7 = geom.addLine(p5, p10)
    l8 = geom.addLine(p4, p9)

    p11 = geom.addPoint(R_mid, 0., - l_pp - l_b, lc_2)

    p12 = geom.addPoint(R_mid + h_b, 0., - l_pp - l_b, lc_2)
    p13 = geom.addPoint(R_mid, h_b, - l_pp - l_b, lc_2)
    p14 = geom.addPoint(R_mid - h_b, 0., - l_pp - l_b, lc_2)

    l9 = geom.addLine(p7, p14)
    l10 = geom.addLine(p14, p12)
    l11 = geom.addLine(p12, p8)
    l12 = geom.addCircleArc(p8, p6, p10)
    l13 = geom.addLine(p10, p9)
    l14 = geom.addCircleArc(p9, p6, p7)

    l15 = geom.addCircleArc(p12, p11, p13)
    l16 = geom.addCircleArc(p13, p11, p14)

    ll2 = geom.addCurveLoop([-l11, -l10, -l9, -l5, l1, l6])
    s2 = geom.addPlaneSurface([ll2])

    ll3 = geom.addCurveLoop([-l6, l2, l7, -l12])
    s3 = geom.addSurfaceFilling([ll3])

    ll4 = geom.addCurveLoop([-l13, -l7, l3, l8])
    s4 = geom.addPlaneSurface([ll4])

    ll5 = geom.addCurveLoop([l5, -l14, -l8, l4])
    s5 = geom.addSurfaceFilling([ll5])

    ll6 = geom.addCurveLoop([l9, -l16, -l15, l11, l12, l13, l14])
    s6 = geom.addPlaneSurface([ll6])

    ll7 = geom.addCurveLoop([l10, l15, l16])
    s7 = geom.addPlaneSurface([ll7])

    sl1 = geom.addSurfaceLoop([s1, s2, s3, s4, s5, s6, s7])
    vol1 = geom.addVolume([sl1])

    # ________________________________________________________________________________
    # BURNER

    p15 = geom.addPoint(R_mid, 0., - l_pp, lc_2)

    p16 = geom.addPoint(R_mid + h_b, 0., - l_pp, lc_2)
    p17 = geom.addPoint(R_mid, h_b, - l_pp, lc_2)
    p18 = geom.addPoint(R_mid - h_b, 0., - l_pp, lc_2)

    p19 = geom.addPoint(R_mid + h_pp, 0., - l_pp, lc_2)
    p20 = geom.addPoint(R_mid, h_pp, - l_pp, lc_2)
    p21 = geom.addPoint(R_mid - h_pp, 0., - l_pp, lc_2)

    l17 = geom.addLine(p14, p18)
    l18 = geom.addLine(p12, p16)
    l19 = geom.addLine(p13, p17)

    l20 = geom.addLine(p18, p21)
    l21 = geom.addLine(p21, p19)
    l22 = geom.addLine(p19, p16)

    l23 = geom.addCircleArc(p16, p15, p17)
    l24 = geom.addCircleArc(p17, p15, p18)

    l25 = geom.addCircleArc(p19, p15, p20)
    l26 = geom.addCircleArc(p20, p15, p21)

    ll8 = geom.addCurveLoop([-l22, -l21, -l20, -l17, l10, l18])
    s8 = geom.addPlaneSurface([ll8])

    ll9 = geom.addCurveLoop([-l23, -l18, l15, l19])
    s9 = geom.addSurfaceFilling([ll9])

    ll10 = geom.addCurveLoop([-l24, -l19, l16, l17])
    s10 = geom.addSurfaceFilling([ll10])

    ll11 = geom.addCurveLoop([l20, -l26, -l25, l22, l23, l24])
    s11 = geom.addPlaneSurface([ll11])

    ll12 = geom.addCurveLoop([l21, l25, l26])
    s12 = geom.addPlaneSurface([ll12])

    sl2 = geom.addSurfaceLoop([s7, s8, s9, s10, s11, s12])
    vol2 = geom.addVolume([sl2])

    # ________________________________________________________________________________
    # INJECTION

    p22 = geom.addPoint(R_mid, 0., 0., lc_2)

    p23 = geom.addPoint(R_mid + h_pp, 0., 0., lc_2)
    p24 = geom.addPoint(R_mid, h_pp, 0., lc_2)
    p25 = geom.addPoint(R_mid - h_pp, 0., 0., lc_2)

    l27 = geom.addLine(p21, p25)
    l28 = geom.addLine(p19, p23)
    l29 = geom.addLine(p20, p24)

    l30 = geom.addLine(p25, p23)
    l31 = geom.addCircleArc(p23, p22, p24)
    l32 = geom.addCircleArc(p24, p22, p25)

    ll13 = geom.addCurveLoop([-l30, -l27, l21, l28])
    s13 = geom.addPlaneSurface([ll13])

    ll14 = geom.addCurveLoop([-l31, -l28, l25, l29])
    s14 = geom.addSurfaceFilling([ll14])

    ll15 = geom.addCurveLoop([-l32, -l29, l26, l27])
    s15 = geom.addSurfaceFilling([ll15])

    ll16 = geom.addCurveLoop([l30, l31, l32])
    s16 = geom.addPlaneSurface([ll16])

    sl3 = geom.addSurfaceLoop([s12, s13, s14, s15, s16])
    vol3 = geom.addVolume([sl3])

    # ________________________________________________________________________________
    # FLAME

    p26 = geom.addPoint(R_mid + h_f, 0., 0., lc_2)
    p27 = geom.addPoint(R_mid, h_f, 0., lc_2)
    p28 = geom.addPoint(R_mid - h_f, 0., 0., lc_2)

    l33 = geom.addLine(p28, p25)
    l34 = geom.addLine(p23, p26)
    l35 = geom.addCircleArc(p26, p22, p27)
    l36 = geom.addCircleArc(p27, p22, p28)

    ll17 = geom.addCurveLoop([-l36, -l35, -l34, l31, l32, -l33])
    s17 = geom.addPlaneSurface([ll17])

    p29 = geom.addPoint(R_mid, 0., 0. + l_f, lc_2)

    p30 = geom.addPoint(R_mid + h_f, 0., 0. + l_f, lc_2)
    p31 = geom.addPoint(R_mid, h_f, 0. + l_f, lc_2)
    p32 = geom.addPoint(R_mid - h_f, 0., 0. + l_f, lc_2)

    l37 = geom.addLine(p28, p32)
    l38 = geom.addLine(p26, p30)
    l39 = geom.addLine(p27, p31)

    l40 = geom.addLine(p32, p30)
    l41 = geom.addCircleArc(p30, p29, p31)
    l42 = geom.addCircleArc(p31, p29, p32)

    ll18 = geom.addCurveLoop([-l40, -l37, l33, l30, l34, l38])
    s18 = geom.addPlaneSurface([ll18])

    ll19 = geom.addCurveLoop([-l41, -l38, l35, l39])
    s19 = geom.addSurfaceFilling([ll19])

    ll20 = geom.addCurveLoop([-l42, -l39, l36, l37])
    s20 = geom.addSurfaceFilling([ll20])

    ll21 = geom.addCurveLoop([l40, l41, l42])
    s21 = geom.addPlaneSurface([ll21])

    sl4 = geom.addSurfaceLoop([s16, s17, s18, s19, s20, s21])
    vol4 = geom.addVolume([sl4])

    # ________________________________________________________________________________
    # COMBUSTION CHAMBER

    p33 = geom.addPoint(0., 0., 0., lc_1)

    p34 = geom.addPoint(R_in_cc, 0., 0., lc_1)
    p35 = geom.addPoint(R_out_cc, 0., 0., lc_1)
    p36 = geom.addPoint(R_in_cc * cos(theta), R_in_cc * sin(theta), 0., lc_1)
    p37 = geom.addPoint(R_out_cc * cos(theta), R_out_cc * sin(theta), 0., lc_1)

    l43 = geom.addLine(p34, p28)
    l44 = geom.addLine(p26, p35)

    l45 = geom.addCircleArc(p35, p33, p37)
    l46 = geom.addLine(p37, p36)
    l47 = geom.addCircleArc(p36, p33, p34)

    ll22 = geom.addCurveLoop([-l44, l35, l36, -l43, -l47, -l46, -l45])
    s22 = geom.addPlaneSurface([ll22])

    p38 = geom.addPoint(0., 0., l_cc + l_ec, lc_1)
    p39 = geom.addPoint(R_in_cc, 0., l_cc + l_ec, lc_1)
    p40 = geom.addPoint(R_out_cc, 0., l_cc + l_ec, lc_1)
    p41 = geom.addPoint(R_in_cc * cos(theta), R_in_cc * sin(theta), l_cc + l_ec, lc_1)
    p42 = geom.addPoint(R_out_cc * cos(theta), R_out_cc * sin(theta), l_cc + l_ec, lc_1)

    l48 = geom.addLine(p34, p39)
    l49 = geom.addLine(p35, p40)
    l50 = geom.addLine(p37, p42)
    l51 = geom.addLine(p36, p41)

    l52 = geom.addLine(p39, p40)
    l53 = geom.addCircleArc(p40, p38, p42)
    l54 = geom.addLine(p42, p41)
    l55 = geom.addCircleArc(p41, p38, p39)

    ll23 = geom.addCurveLoop([-l52, -l48, l43, l37, l40, -l38, l44, l49])
    s23 = geom.addPlaneSurface([ll23])

    ll24 = geom.addCurveLoop([-l49, l45, l50, -l53])
    s24 = geom.addSurfaceFilling([ll24])

    ll25 = geom.addCurveLoop([-l54, -l50, l46, l51])
    s25 = geom.addPlaneSurface([ll25])

    ll26 = geom.addCurveLoop([l48, -l55, -l51, l47])
    s26 = geom.addSurfaceFilling([ll26])

    ll27 = geom.addCurveLoop([l52, l53, l54, l55])
    s27 = geom.addPlaneSurface([ll27])

    sl5 = geom.addSurfaceLoop([s19, s20, s21, s22, s23, s24, s25, s26, s27])
    vol5 = geom.addVolume([sl5])


def apply_symmetry():

    symmetry = gmsh.model.geo.symmetrize

    gmsh.model.geo.synchronize()
    a = gmsh.model.getEntities(dim=2) # all defined surfaces
    symmetry(gmsh.model.geo.copy(a), 0, 1, 0, 0) #mirror geom wrt y axis

    # gmsh.model.geo.synchronize()
    my_vol = gmsh.model.getEntities(dim=3)
    symmetry(gmsh.model.geo.copy(my_vol), 0, 1, 0, 0) #mirror geom wrt y axis

    gmsh.model.geo.synchronize()
    b = gmsh.model.getEntities(dim=2)

    tmp = set(a)
    b = [x for x in b if x not in tmp] #Substract elements of a from b
    
    #Substract the surfaces which lies on xz plane that should not be reflected.
    indices = [2, 8, 13, 18, 23]  # not reflected surfaces
    indices = [x - 1 for x in indices]
    for index in sorted(indices, reverse=True):
        a.pop(index)
        
    # Get the surface tags
    a = [x[1] for x in a]
    b = [x[1] for x in b]
    # Assign the same tags of half geometry to the reflected geometry, 
    # the new tags b is changed to tags a again
    gmsh.model.mesh.setPeriodic(2, b, a, reflection_matrix())

    
def apply_rotation():

    theta = 22.5 / 2
    theta *= pi / 180

    rotate = gmsh.model.geo.rotate

    gmsh.model.geo.synchronize()
    my_surf = gmsh.model.getEntities(dim=2)
    my_surf_old = gmsh.model.getEntities(dim=2)

    a = gmsh.model.getEntities(dim=2)  # for periodicity

    indices = [29, 46]  # the indices of surfaces not to be rotated
    for index in sorted(indices, reverse=True):
        a.pop(index)

    a = [x[1] for x in a] # get surface tags

    my_vol = gmsh.model.getEntities(dim=3)

    for i in range(1, 16):
        # Updates surface tags for every rotation

        angle = 2 * i * theta

        rotate(gmsh.model.geo.copy(my_surf), 0, 0, 0, 0, 0, 1, angle)

        gmsh.model.geo.synchronize()
        my_surf_new = gmsh.model.getEntities(dim=2)

        tmp = set(my_surf_old)
        b = [x for x in my_surf_new if x not in tmp]
        b = [x[1] for x in b]
        
        rotate(gmsh.model.geo.copy(my_vol), 0, 0, 0, 0, 0, 1, angle)

        if i == 15:
            indices = [4, 25]
            indices = [x - 1 for x in indices]
            for index in sorted(indices, reverse=True):
                a.pop(index)

        gmsh.model.mesh.setPeriodic(2, b, a, rotation_matrix(angle))

        my_surf_old = my_surf_new


def add_physical_entities():

    my_dict = physical.physical_groups()

    gmsh.model.geo.synchronize()
    my_surf = gmsh.model.getEntities(dim=2)

    # gmsh.model.addPhysicalGroup(2, [x[1] for x in my_surf], tag=99)

    def my_func(my_list_1, my_tag):
        # my_surf lives in the enclosing scope
        my_list_2 = [my_surf[x][1] for x in my_list_1]
        gmsh.model.addPhysicalGroup(2, my_list_2, my_tag)

    my_func(my_dict['pl_rear'], 1)
    my_func(my_dict['pl_top'], 2)
    my_func(my_dict['pl_bottom'], 3)
    my_func(my_dict['pl_front'], 4)
    my_func(my_dict['b_lateral'], 5)
    my_func(my_dict['b_front'], 6)
    my_func(my_dict['pp_lateral'], 7)
    my_func(my_dict['cc_rear'], 8)
    my_func(my_dict['cc_top'], 9)
    my_func(my_dict['cc_bottom'], 10)
    my_func(my_dict['cc_front'], 11)

    my_vol = gmsh.model.getEntities(dim=3)

    for i in range(16):
        my_list = [my_vol[x][1] for x in [3 + i * 10, 8 + i * 10]]
        gmsh.model.addPhysicalGroup(3, my_list, tag=i)

    my_list = []
    for i in range(16):
        my_list += [3 + i * 10, 8 + i * 10]
    my_list = [my_vol[x] for x in my_list]

    gmsh.model.addPhysicalGroup(3, [x[1] for x in my_vol if x not in my_list], tag=99)


def fltk_options():

    # Type of entity label (0: description,
    #                       1: elementary entity tag,
    #                       2: physical group tag)
    gmsh.option.setNumber("Geometry.LabelType", 2)

    gmsh.option.setNumber("Geometry.PointNumbers", 0)
    gmsh.option.setNumber("Geometry.LineNumbers", 0)
    gmsh.option.setNumber("Geometry.SurfaceNumbers", 2)
    gmsh.option.setNumber("Geometry.VolumeNumbers", 0)

    # Mesh coloring(0: by element type, 1: by elementary entity,
    #                                   2: by physical group,
    #                                   3: by mesh partition)
    gmsh.option.setNumber("Mesh.ColorCarousel", 2)

    gmsh.option.setNumber("Mesh.SurfaceEdges", 0)
    gmsh.option.setNumber("Mesh.SurfaceFaces", 0)

    gmsh.option.setNumber("Mesh.VolumeEdges", 0)
    gmsh.option.setNumber("Mesh.VolumeFaces", 0)

    
    # PHYSICAL LABELS
    # # Type of entity label (0: description,
    # #                       1: elementary entity tag,
    # #                       2: physical group tag)
    # gmsh.option.setNumber("Geometry.LabelType", 2)

    # gmsh.option.setNumber("Geometry.PointNumbers", 0)
    # gmsh.option.setNumber("Geometry.LineNumbers", 0)
    # gmsh.option.setNumber("Geometry.SurfaceNumbers", 2)
    # gmsh.option.setNumber("Geometry.VolumeNumbers", 0)

    # # Mesh coloring(0: by element type, 1: by elementary entity,
    # #                                   2: by physical group,
    # #                                   3: by mesh partition)
    # gmsh.option.setNumber("Mesh.ColorCarousel", 2)

    # gmsh.option.setNumber("Mesh.SurfaceEdges", 0)
    # gmsh.option.setNumber("Mesh.SurfaceFaces", 0)

    # gmsh.option.setNumber("Mesh.VolumeEdges", 0)
    # gmsh.option.setNumber("Mesh.VolumeFaces", 0)

if __name__ == '__main__':

    geom_1('MeshDir/micca_flame', fltk=True, l_ec=0.041)
