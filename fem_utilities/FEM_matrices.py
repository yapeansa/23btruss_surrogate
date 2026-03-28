import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

def static_analysis(nodes):
    dof = 3
    maxdof = dof*np.size(nodes,0)
    F = np.zeros((maxdof,1))
    restr = np.zeros(dof*np.size(nodes,0))

    for i in range(np.size(nodes,0)):
        RX = nodes[i,2]
        RY = nodes[i,3]
        RZ = nodes[i,4]
        Fx = nodes[i,5]
        Fy = nodes[i,6]
        Mz = nodes[i,7]

        gl1 = int(dof*(i+1)-2)
        gl2 = int(dof*(i+1)-1)
        gl3 = int(dof*(i+1))

        if RX == 1:
            restr[gl1-1] = 1
        if RY == 1:
            restr[gl2-1] = 1
        if RZ == 1:
            restr[gl3-1] = 1
        if Fx != 0:
            F[gl1-1] = Fx
        if Fy != 0:
            F[gl2-1] = Fy
        if Mz != 0:
            F[gl3-1] = Mz

    ccnt = restr==1
    ccnt = ccnt.nonzero()
    ccnt = np.asarray(ccnt)
    Fr = np.delete(F,ccnt, axis = 0)

    return Fr, restr

def FEM_matrices(nodes, bars):
    dof = 3

    # Elements properties
    L = np.zeros(np.size(bars,0))
    seno = np.zeros(np.size(bars,0))
    coss = np.zeros(np.size(bars,0))
    for i in range(np.size(bars,0)):
        N1 = int(bars[i,0])
        N2 = int(bars[i,1])

        x1 = nodes[N1-1,0]
        y1 = nodes[N1-1,1]
        x2 = nodes[N2-1,0]
        y2 = nodes[N2-1,1]

        Lx = x2 - x1
        Ly = y2 - y1

        Ls = np.sqrt(Lx**2 + Ly**2)
        L[i] = Ls
        seno[i]  = Ly/Ls
        coss[i]  = Lx/Ls

    # Matrices
    maxdof = dof*np.size(nodes,0)

    K = np.zeros((maxdof,maxdof))
    M = np.zeros((maxdof,maxdof))

    for i in range(np.size(bars,0)):
        Ls = L[i]
        sen = seno[i]
        cos = coss[i]
        N1 = bars[i,0]
        N2 = bars[i,1]
        A = bars[i,2]
        E = bars[i,3]
        I = bars[i,4]
        ro = bars[i,5]

        K11 = (E*A)/Ls
        K22 = (12*E*I)/(Ls**3)
        K32 = (6*E*I)/(Ls**2)
        K23 = K32
        K33 = (4*E*I)/Ls
        K36 = (2*E*I)/Ls

        K1 = np.array([[ K11,    0,     0, -K11,    0,     0],
                        [    0,  K22,   K23,    0, -K22,   K32],
                        [    0,  K32,   K33,    0, -K32,   K36],
                        [ -K11,    0,     0,  K11,    0,     0],
                        [    0, -K22,  -K23,    0,  K22,  -K32],
                        [    0,  K32,   K36,    0, -K32,   K33]])

        M1 = ((ro*A*Ls)/420)*np.array([[140,      0,       0,  70,      0,       0],
                                        [  0,    156,   22*Ls,   0,     54,  -13*Ls],
                                        [  0,  22*Ls,  4*Ls**2,   0,  13*Ls, -3*Ls**2],
                                        [ 70,      0,       0, 140,      0,       0],
                                        [  0,     54,   13*Ls,   0,    156,  -22*Ls],
                                        [  0, -13*Ls, -3*Ls**2,   0, -22*Ls,  4*Ls**2]])

        Rot = np.array([[ cos,  sen,    0,    0,   0,   0],          # Rotation matrix
                        [-sen,  cos,    0,    0,   0,   0],
                        [   0,    0,    1,    0,   0,   0],
                        [   0,    0,    0,  cos, sen,   0],
                        [   0,    0,    0, -sen, cos,   0],
                        [   0,    0,    0,    0,   0,   1]])

        Klr = np.dot(np.dot(Rot.T, K1), Rot)
        Mrr = np.dot(np.dot(Rot.T, M1), Rot)

        gl1 = int(dof*N1-2)
        gl2 = int(dof*N1-1)
        gl3 = int(dof*N1)
        gl4 = int(dof*N2-2)
        gl5 = int(dof*N2-1)
        gl6 = int(dof*N2)

        K[gl1-1:gl3, gl1-1:gl3] += Klr[0:3, 0:3]
        K[gl4-1:gl6, gl1-1:gl3] += Klr[3:6, 0:3]
        K[gl1-1:gl3, gl4-1:gl6] += Klr[0:3, 3:6]
        K[gl4-1:gl6, gl4-1:gl6] += Klr[3:6, 3:6]

        M[gl1-1:gl3, gl1-1:gl3] += Mrr[0:3, 0:3]
        M[gl4-1:gl6, gl1-1:gl3] += Mrr[3:6, 0:3]
        M[gl1-1:gl3, gl4-1:gl6] += Mrr[0:3, 3:6]
        M[gl4-1:gl6, gl4-1:gl6] += Mrr[3:6, 3:6]

    # Boundary conditions
    restr = np.zeros(dof*np.size(nodes,0))
    for i in range(np.size(nodes,0)):
        RX = nodes[i,2]
        RY = nodes[i,3]
        RZ = nodes[i,4]

        gl1 = int(dof*(i+1)-2)
        gl2 = int(dof*(i+1)-1)
        gl3 = int(dof*(i+1))

        if RX == 1:
            restr[gl1-1] = 1
        if RY == 1:
            restr[gl2-1] = 1
        if RZ == 1:
            restr[gl3-1] = 1

    ccnt = restr==1
    ccnt = ccnt.nonzero()
    Kr = np.delete(K,ccnt, axis = 0)
    Kr = np.delete(Kr,ccnt, axis = 1)
    Mr = np.delete(M,ccnt, axis = 0)
    Mr = np.delete(Mr,ccnt, axis = 1)

    return K, M, Kr, Mr

def run_simulation(nodes, bars):
    K, _, Kr, _ = FEM_matrices(nodes, bars)
    Fr, restr   = static_analysis(nodes)
    return Kr, Fr, restr

def solver(nodes, bars):
    dof = 3
    maxdof = dof*np.size(nodes, 0)
    Kr, Fr, restr = run_simulation(nodes, bars)
    Kr_sparse = csc_matrix(Kr)
    disp_r = spsolve(Kr_sparse, Fr)
    disp = np.zeros((maxdof, 1))
    ccnt = restr==0
    ccnt = ccnt.nonzero()
    disp[ccnt, 0] = disp_r.T

    # forces = K@disp

    return disp
