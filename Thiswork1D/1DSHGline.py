import gc
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv


try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    MPI_RANK = COMM.Get_rank()
    MPI_SIZE = COMM.Get_size()
except Exception:
    MPI = None
    COMM = None
    MPI_RANK = 0
    MPI_SIZE = 1


FIXED_BETA  = 0.9
Na          = 100
ALPHA_MIN, ALPHA_MAX = 0.0, 2.0


Nkx, Nky = 2000, 2000
KX_MIN, KX_MAX = -np.pi, np.pi
KY_MIN, KY_MAX =  0.0, 2.0*np.pi


OMEGA   = 5
ETA_PV1 = 0.01
ETA_PV2 = 0.01


OUTDIR    = "SHG_1D_Scan_Alpha_Output"


USE_WILSON_LOOP = True
PARALLEL        = True
N_WORKERS       = None


KX_CHUNKS_PER_ALPHA = None


USE_LOW_MEMORY_BLOCK_INTEGRATION = True
MEMORY_CLEAN_EVERY = 20


def H_from_d(d):
    d1, d2, d3 = d[...,0], d[...,1], d[...,2]
    H = np.empty(d.shape[:-1] + (2,2), dtype=complex)
    H[...,0,0] = d3
    H[...,0,1] = d1 - 1j*d2
    H[...,1,0] = d1 + 1j*d2
    H[...,1,1] = -d3
    return H

def d_components(kx, ky, alpha, beta):
    s, c = np.sin, np.cos
    d1 = s(kx)*c(ky) + alpha*s(kx) + beta*s(2*kx)
    d2 = s(kx)*s(ky)
    d3 = c(kx)
    return np.stack([d1, d2, d3], axis=-1)

def d_derivs(kx, ky, alpha, beta):
    s, c = np.sin, np.cos
    ax1 = c(kx)*c(ky) + alpha*c(kx) + 2*beta*c(2*kx)
    ax2 = c(kx)*s(ky)
    ax3 = -s(kx)
    a_x = np.stack([ax1, ax2, ax3], axis=-1)

    ay1 = -s(kx)*s(ky)
    ay2 =  s(kx)*c(ky)
    ay3 =  0.0
    a_y = np.stack([ay1, ay2, np.full_like(ay1, ay3)], axis=-1)
    return a_x, a_y

def V_from_a(a):
    a1, a2, a3 = a[...,0], a[...,1], a[...,2]
    V = np.empty(a.shape[:-1] + (2,2), dtype=complex)
    V[...,0,0] = a3
    V[...,0,1] = a1 - 1j*a2
    V[...,1,0] = a1 + 1j*a2
    V[...,1,1] = -a3
    return V

def kernel_plus(x, eta): return 1.0 / (x + 1j*eta/2.0)
def kernel_minus(x, eta): return 1.0 / (x - 1j*eta/2.0)

def eigh_2x2_herm_batch(H): return np.linalg.eigh(H)

def mat_elem(U_left, Op, U_right):
    left  = np.conj(U_left)[..., None, :]
    right = U_right[..., :, None]
    return (left @ Op @ right)[..., 0, 0]

def band_vectors(U, band_idx): return U[..., :, band_idx]
def overlap(psi, phi): return np.sum(np.conj(psi) * phi, axis=-1)

def roll_plus(arr, axis): return np.roll(arr, -1, axis=axis)
def roll_minus(arr, axis): return np.roll(arr, +1, axis=axis)

def d_arg_central(z, axis, dk):

    z_plus  = roll_plus(z, axis)
    z_minus = roll_minus(z, axis)
    up = z_plus  / np.abs(z_plus)
    um = z_minus / np.abs(z_minus)
    return np.imag(np.log(up * np.conj(um) + 0j)) / (2.0*dk)

def d_logabs_central(z, axis, dk):

    z_plus  = roll_plus(z, axis)
    z_minus = roll_minus(z, axis)
    return (np.log(np.abs(z_plus)) - np.log(np.abs(z_minus))) / (2.0*dk)

def berry_connection_forward(u, axis, dk):
    u_plus = roll_plus(u, axis)
    link = overlap(u, u_plus)
    return np.imag(np.log(link + 0j)) / dk

def berry_connection_central(u, axis, dk):
    A_f = berry_connection_forward(u, axis, dk)
    A_b = -berry_connection_forward(roll_minus(u, axis), axis, dk)
    return 0.5*(A_f + A_b)

def r_from_v(v_cv, omega_cv):

    return v_cv / (1j * omega_cv)

def R_from_definition(vb_cv, uc, uv, dk_a, axis_a):
    dphi = d_arg_central(vb_cv, axis=axis_a, dk=dk_a)
    A_c  = berry_connection_central(uc, axis=axis_a, dk=dk_a)
    A_v  = berry_connection_central(uv, axis=axis_a, dk=dk_a)
    return -dphi + (A_c - A_v)

def R_from_wilson(uc, uv, rb_cv, dk_a, axis_a):
    uc_p = roll_plus(uc, axis_a)
    uv_p = roll_plus(uv, axis_a)
    rb_p = roll_plus(rb_cv, axis_a)
    Wp = overlap(uc, uc_p) * rb_p * overlap(uv_p, uv) * np.conj(rb_cv)
    uc_m = roll_minus(uc, axis_a)
    uv_m = roll_minus(uv, axis_a)
    rb_m = roll_minus(rb_cv, axis_a)
    Wm = overlap(uc, uc_m) * rb_m * overlap(uv_m, uv) * np.conj(rb_cv)
    return - np.imag(np.log(Wp/(Wm+0j) + 0j)) / (2.0*dk_a)

def subterms_on_kgrid_numeric(kx, ky, alpha, beta):

    d = d_components(kx, ky, alpha, beta)
    a_x, a_y = d_derivs(kx, ky, alpha, beta)
    H  = H_from_d(d)
    Vx = V_from_a(a_x)
    Vy = V_from_a(a_y)

    E, U = eigh_2x2_herm_batch(H)
    uv = band_vectors(U, 0)
    uc = band_vectors(U, 1)
    Ev = E[...,0]
    Ec = E[...,1]
    omega_cv = Ec - Ev

    v_x_12 = mat_elem(uv, Vx, uc)
    v_y_12 = mat_elem(uv, Vy, uc)
    v_x_21 = np.conj(v_x_12)
    v_y_21 = np.conj(v_y_12)

    v_x_22 = mat_elem(uc, Vx, uc).real
    v_x_11 = mat_elem(uv, Vx, uv).real


    ratio  = (v_x_22 - v_x_11) / omega_cv

    Nkx = kx.shape[0]
    dkx = (KX_MAX - KX_MIN) / Nkx
    dky = (KY_MAX - KY_MIN) / Nky

    dln_abs_vx = d_logabs_central(v_x_12, axis=0, dk=dkx)

    if USE_WILSON_LOOP:
        r_x_12 = r_from_v(v_x_12, omega_cv)
        R_xx_12 = R_from_wilson(uv, uc, r_x_12, dk_a=dkx, axis_a=0)
        R_yx12  = R_from_wilson(uv, uc, r_x_12, dk_a=dky, axis_a=1)
    else:
        R_xx_12 = R_from_definition(v_x_12, uv, uc, dk_a=dkx, axis_a=0)
        R_yx12  = R_from_definition(v_x_12, uv, uc, dk_a=dky, axis_a=1)

    R_yx21 = - R_yx12


    x1_1 = (OMEGA - omega_cv)/2.0
    x1_2 = (-OMEGA - omega_cv)/2.0
    x2_1 = (2.0*OMEGA - omega_cv)/2.0
    x2_2 = (-2.0*OMEGA - omega_cv)/2.0

    K1_1 = kernel_plus(x1_1, ETA_PV1)
    K1_2 = kernel_minus(x1_2, ETA_PV1)
    K2_1 = kernel_plus(x2_1, ETA_PV2)
    K2_2 = kernel_minus(x2_2, ETA_PV2)

    axpn2 = np.abs(v_x_12)**2
    v_y12_x21 = v_y_12 * v_x_21
    v_y21_x12 = v_y_21 * v_x_12

    factor_T1a = -1j * axpn2 * R_yx21
    T1a_1_k = factor_T1a * K1_1
    T1a_2_k = factor_T1a * (-1.0) * K1_2

    T1b_1_k = (-1.0 * v_y12_x21 * ratio) * K1_1
    T1b_2_k = (-1.0 * v_y21_x12 * ratio) * K1_2

    T2a_1_k = (-1j * v_y12_x21 * R_xx_12) * K2_1
    T2a_2_k = (+1j * v_y21_x12 * R_xx_12) * K2_2

    T2b_1_k = (-1.0 * v_y12_x21 * dln_abs_vx) * K2_1
    T2b_2_k = (-1.0 * v_y21_x12 * dln_abs_vx) * K2_2

    T2c_1_k = (-1.0 * v_y12_x21 * ratio) * K2_1
    T2c_2_k = (-1.0 * v_y21_x12 * ratio) * K2_2


    return (T1a_1_k, T1a_2_k, T1b_1_k, T1b_2_k,
            T2a_1_k, T2a_2_k, T2b_1_k, T2b_2_k, T2c_1_k, T2c_2_k)


def subterms_on_kgrid_numeric_with_dk(kx, ky, alpha, beta, dkx, dky):
    d = d_components(kx, ky, alpha, beta)
    a_x, a_y = d_derivs(kx, ky, alpha, beta)
    H  = H_from_d(d)
    Vx = V_from_a(a_x)
    Vy = V_from_a(a_y)

    E, U = eigh_2x2_herm_batch(H)
    uv = band_vectors(U, 0)
    uc = band_vectors(U, 1)
    Ev = E[...,0]
    Ec = E[...,1]
    omega_cv = Ec - Ev

    v_x_12 = mat_elem(uv, Vx, uc)
    v_y_12 = mat_elem(uv, Vy, uc)
    v_x_21 = np.conj(v_x_12)
    v_y_21 = np.conj(v_y_12)

    v_x_22 = mat_elem(uc, Vx, uc).real
    v_x_11 = mat_elem(uv, Vx, uv).real


    ratio  = (v_x_22 - v_x_11) / omega_cv

    dln_abs_vx = d_logabs_central(v_x_12, axis=0, dk=dkx)

    if USE_WILSON_LOOP:
        r_x_12 = r_from_v(v_x_12, omega_cv)
        R_xx_12 = R_from_wilson(uv, uc, r_x_12, dk_a=dkx, axis_a=0)
        R_yx12  = R_from_wilson(uv, uc, r_x_12, dk_a=dky, axis_a=1)
    else:
        R_xx_12 = R_from_definition(v_x_12, uv, uc, dk_a=dkx, axis_a=0)
        R_yx12  = R_from_definition(v_x_12, uv, uc, dk_a=dky, axis_a=1)

    R_yx21 = - R_yx12


    x1_1 = (OMEGA - omega_cv)/2.0
    x1_2 = (-OMEGA - omega_cv)/2.0
    x2_1 = (2.0*OMEGA - omega_cv)/2.0
    x2_2 = (-2.0*OMEGA - omega_cv)/2.0

    K1_1 = kernel_plus(x1_1, ETA_PV1)
    K1_2 = kernel_minus(x1_2, ETA_PV1)
    K2_1 = kernel_plus(x2_1, ETA_PV2)
    K2_2 = kernel_minus(x2_2, ETA_PV2)

    axpn2 = np.abs(v_x_12)**2
    v_y12_x21 = v_y_12 * v_x_21
    v_y21_x12 = v_y_21 * v_x_12

    factor_T1a = -1j * axpn2 * R_yx21
    T1a_1_k = factor_T1a * K1_1
    T1a_2_k = factor_T1a * (-1.0) * K1_2

    T1b_1_k = (-1.0 * v_y12_x21 * ratio) * K1_1
    T1b_2_k = (-1.0 * v_y21_x12 * ratio) * K1_2

    T2a_1_k = (-1j * v_y12_x21 * R_xx_12) * K2_1
    T2a_2_k = (+1j * v_y21_x12 * R_xx_12) * K2_2

    T2b_1_k = (-1.0 * v_y12_x21 * dln_abs_vx) * K2_1
    T2b_2_k = (-1.0 * v_y21_x12 * dln_abs_vx) * K2_2

    T2c_1_k = (-1.0 * v_y12_x21 * ratio) * K2_1
    T2c_2_k = (-1.0 * v_y21_x12 * ratio) * K2_2


    return (T1a_1_k, T1a_2_k, T1b_1_k, T1b_2_k,
            T2a_1_k, T2a_2_k, T2b_1_k, T2b_2_k, T2c_1_k, T2c_2_k)


def integrate_kgrid_block_numeric_lowmem(kx, ky, alpha, beta, dkx, dky, core):
    d = d_components(kx, ky, alpha, beta)
    a_x, a_y = d_derivs(kx, ky, alpha, beta)
    H  = H_from_d(d)
    Vx = V_from_a(a_x)
    Vy = V_from_a(a_y)
    del d, a_x, a_y

    E, U = eigh_2x2_herm_batch(H)
    del H

    uv = band_vectors(U, 0)
    uc = band_vectors(U, 1)
    Ev = E[..., 0]
    Ec = E[..., 1]
    omega_cv = Ec - Ev
    del E, Ev, Ec

    v_x_12 = mat_elem(uv, Vx, uc)
    v_y_12 = mat_elem(uv, Vy, uc)
    v_x_21 = np.conj(v_x_12)
    v_y_21 = np.conj(v_y_12)

    v_x_22 = mat_elem(uc, Vx, uc).real
    v_x_11 = mat_elem(uv, Vx, uv).real
    del Vx, Vy


    ratio = (v_x_22 - v_x_11) / omega_cv
    del v_x_22, v_x_11

    dln_abs_vx = d_logabs_central(v_x_12, axis=0, dk=dkx)

    if USE_WILSON_LOOP:
        r_x_12 = r_from_v(v_x_12, omega_cv)
        R_xx_12 = R_from_wilson(uv, uc, r_x_12, dk_a=dkx, axis_a=0)
        R_yx12  = R_from_wilson(uv, uc, r_x_12, dk_a=dky, axis_a=1)
        del r_x_12
    else:
        R_xx_12 = R_from_definition(v_x_12, uv, uc, dk_a=dkx, axis_a=0)
        R_yx12  = R_from_definition(v_x_12, uv, uc, dk_a=dky, axis_a=1)

    del U, uv, uc
    R_yx21 = -R_yx12
    del R_yx12


    omega_c = omega_cv[core, :]
    ratio_c = ratio[core, :]
    vx12_c  = v_x_12[core, :]
    vy12_c  = v_y_12[core, :]
    vx21_c  = v_x_21[core, :]
    vy21_c  = v_y_21[core, :]
    Rxx_c   = R_xx_12[core, :]
    Ryx21_c = R_yx21[core, :]
    dln_c   = dln_abs_vx[core, :]


    x1_1 = (OMEGA - omega_c) / 2.0
    x1_2 = (-OMEGA - omega_c) / 2.0
    x2_1 = (2.0 * OMEGA - omega_c) / 2.0
    x2_2 = (-2.0 * OMEGA - omega_c) / 2.0

    K1_1 = kernel_plus(x1_1, ETA_PV1)
    K1_2 = kernel_minus(x1_2, ETA_PV1)
    K2_1 = kernel_plus(x2_1, ETA_PV2)
    K2_2 = kernel_minus(x2_2, ETA_PV2)
    del x1_1, x1_2, x2_1, x2_2

    dA = dkx * dky
    pref1 = (1j) / (4.0 * 4 * np.pi**2 * OMEGA**3)
    pref2 = (1j) / (8.0 * 4 * np.pi**2 * OMEGA**3)

    partial = np.empty(10, dtype=complex)

    axpn2 = np.abs(vx12_c)**2
    factor_T1a = -1j * axpn2 * Ryx21_c
    partial[0] = pref1 * (np.sum(factor_T1a * K1_1) * dA)
    partial[1] = pref1 * (np.sum(factor_T1a * (-1.0) * K1_2) * dA)
    del axpn2, factor_T1a, Ryx21_c

    v_y12_x21 = vy12_c * vx21_c
    v_y21_x12 = vy21_c * vx12_c

    partial[2] = pref1 * (np.sum((-1.0 * v_y12_x21 * ratio_c) * K1_1) * dA)
    partial[3] = pref1 * (np.sum((-1.0 * v_y21_x12 * ratio_c) * K1_2) * dA)
    del K1_1, K1_2

    partial[4] = pref2 * (np.sum((-1j * v_y12_x21 * Rxx_c) * K2_1) * dA)
    partial[5] = pref2 * (np.sum((+1j * v_y21_x12 * Rxx_c) * K2_2) * dA)
    del Rxx_c

    partial[6] = pref2 * (np.sum((-1.0 * v_y12_x21 * dln_c) * K2_1) * dA)
    partial[7] = pref2 * (np.sum((-1.0 * v_y21_x12 * dln_c) * K2_2) * dA)
    del dln_c

    partial[8] = pref2 * (np.sum((-1.0 * v_y12_x21 * ratio_c) * K2_1) * dA)
    partial[9] = pref2 * (np.sum((-1.0 * v_y21_x12 * ratio_c) * K2_2) * dA)


    del omega_cv, omega_c, ratio, ratio_c
    del v_x_12, v_y_12, v_x_21, v_y_21
    del vx12_c, vy12_c, vx21_c, vy21_c
    del R_xx_12, R_yx21, dln_abs_vx
    del v_y12_x21, v_y21_x12, K2_1, K2_2

    return partial


def _compute_one(ia, alpha, fixed_beta, KX, KY, d_kx, d_ky):

    terms = subterms_on_kgrid_numeric(KX, KY, alpha, fixed_beta)

    dA = d_kx * d_ky
    pref1 = (1j) / (4.0 * 4 * np.pi**2 * OMEGA**3)
    pref2 = (1j) / (8.0 * 4 * np.pi**2 * OMEGA**3)


    I1a_1 = pref1 * (terms[0].sum() * dA)
    I1a_2 = pref1 * (terms[1].sum() * dA)
    I1b_1 = pref1 * (terms[2].sum() * dA)
    I1b_2 = pref1 * (terms[3].sum() * dA)

    I2a_1 = pref2 * (terms[4].sum() * dA)
    I2a_2 = pref2 * (terms[5].sum() * dA)
    I2b_1 = pref2 * (terms[6].sum() * dA)
    I2b_2 = pref2 * (terms[7].sum() * dA)
    I2c_1 = pref2 * (terms[8].sum() * dA)
    I2c_2 = pref2 * (terms[9].sum() * dA)


    return (ia,
            I1a_1, I1a_2, I1b_1, I1b_2,
            I2a_1, I2a_2, I2b_1, I2b_2, I2c_1, I2c_2)


def _make_kx_blocks(nkx, nblocks):
    nblocks = int(max(1, min(nblocks, nkx)))
    edges = np.linspace(0, nkx, nblocks + 1, dtype=int)
    blocks = []
    for i in range(nblocks):
        start, end = int(edges[i]), int(edges[i + 1])
        if end > start:
            blocks.append((start, end))
    return blocks


def _compute_one_kx_block(ia, alpha, fixed_beta, kx_1d, ky_1d, start, end, d_kx, d_ky):
    if end <= start:
        return ia, np.zeros(10, dtype=complex)

    left_halo = (start - 1) % Nkx
    right_halo = end % Nkx
    core_idx = np.arange(start, end, dtype=int)
    kx_idx = np.concatenate(([left_halo], core_idx, [right_halo]))
    kx_halo = kx_1d[kx_idx]


    KX, KY = np.broadcast_arrays(kx_halo[:, None], ky_1d[None, :])

    core = slice(1, 1 + (end - start))

    if USE_LOW_MEMORY_BLOCK_INTEGRATION:
        partial = integrate_kgrid_block_numeric_lowmem(KX, KY, alpha, fixed_beta, d_kx, d_ky, core)
    else:

        terms = subterms_on_kgrid_numeric_with_dk(KX, KY, alpha, fixed_beta, d_kx, d_ky)
        dA = d_kx * d_ky
        pref1 = (1j) / (4.0 * 4 * np.pi**2 * OMEGA**3)
        pref2 = (1j) / (8.0 * 4 * np.pi**2 * OMEGA**3)
        partial = np.empty(10, dtype=complex)
        partial[0] = pref1 * (terms[0][core, :].sum() * dA)
        partial[1] = pref1 * (terms[1][core, :].sum() * dA)
        partial[2] = pref1 * (terms[2][core, :].sum() * dA)
        partial[3] = pref1 * (terms[3][core, :].sum() * dA)
        partial[4] = pref2 * (terms[4][core, :].sum() * dA)
        partial[5] = pref2 * (terms[5][core, :].sum() * dA)
        partial[6] = pref2 * (terms[6][core, :].sum() * dA)
        partial[7] = pref2 * (terms[7][core, :].sum() * dA)
        partial[8] = pref2 * (terms[8][core, :].sum() * dA)
        partial[9] = pref2 * (terms[9][core, :].sum() * dA)
        del terms

    del KX, KY, kx_halo, kx_idx, core_idx
    return ia, partial

def compute_lines_alpha_scan():
    alphas = np.linspace(ALPHA_MIN, ALPHA_MAX, Na)

    keys_sub = ["T1a_1", "T1a_2", "T1b_1", "T1b_2",
                "T2a_1", "T2a_2", "T2b_1", "T2b_2", "T2c_1", "T2c_2"]
    keys_main = ["T1a", "T1b", "T2a", "T2b", "T2c"]

    kx = np.linspace(KX_MIN, KX_MAX, Nkx, endpoint=False)
    ky = np.linspace(KY_MIN, KY_MAX, Nky, endpoint=False)
    d_kx = (KX_MAX - KX_MIN) / Nkx
    d_ky = (KY_MAX - KY_MIN) / Nky

    n_kx_chunks = KX_CHUNKS_PER_ALPHA
    if n_kx_chunks is None:
        n_kx_chunks = MPI_SIZE
    n_kx_chunks = int(max(1, min(n_kx_chunks, Nkx)))
    kx_blocks = _make_kx_blocks(Nkx, n_kx_chunks)


    tasks = [(ia, float(alphas[ia]), start, end)
             for ia in range(Na)
             for (start, end) in kx_blocks]

    local_tasks = tasks[MPI_RANK::MPI_SIZE]
    local_partial = np.zeros((Na, 10), dtype=complex)

    if MPI_RANK == 0:
        pass


    iterator = local_tasks
    if MPI_RANK == 0:
        iterator = local_tasks

    for local_i, (ia, alpha, start, end) in enumerate(iterator):
        ia_ret, partial = _compute_one_kx_block(
            ia, alpha, FIXED_BETA, kx, ky, start, end, d_kx, d_ky
        )
        local_partial[ia_ret, :] += partial

        if MEMORY_CLEAN_EVERY and ((local_i + 1) % MEMORY_CLEAN_EVERY == 0):
            gc.collect()

    if COMM is not None:
        global_partial = np.zeros_like(local_partial) if MPI_RANK == 0 else None
        COMM.Reduce(local_partial, global_partial, op=MPI.SUM, root=0)
    else:
        global_partial = local_partial

    if MPI_RANK != 0:
        return alphas, None

    results = {k: np.zeros(Na, dtype=complex) for k in keys_sub + keys_main + ["Total"]}


    for ia in range(Na):
        i1a1, i1a2, i1b1, i1b2, i2a1, i2a2, i2b1, i2b2, i2c1, i2c2 = global_partial[ia, :]

        results["T1a_1"][ia] = i1a1
        results["T1a_2"][ia] = i1a2
        results["T1b_1"][ia] = i1b1
        results["T1b_2"][ia] = i1b2
        results["T2a_1"][ia] = i2a1
        results["T2a_2"][ia] = i2a2
        results["T2b_1"][ia] = i2b1
        results["T2b_2"][ia] = i2b2
        results["T2c_1"][ia] = i2c1
        results["T2c_2"][ia] = i2c2

        v_t1a = i1a1 + i1a2
        v_t1b = i1b1 + i1b2
        v_t2a = i2a1 + i2a2
        v_t2b = i2b1 + i2b2
        v_t2c = i2c1 + i2c2

        results["T1a"][ia] = v_t1a
        results["T1b"][ia] = v_t1b
        results["T2a"][ia] = v_t2a
        results["T2b"][ia] = v_t2b
        results["T2c"][ia] = v_t2c
        results["Total"][ia] = v_t1a + v_t1b + v_t2a + v_t2b + v_t2c

    return alphas, results


def _ensure_subdirs(base_dir):
    base = Path(base_dir or ".")
    real_dir  = base / "real"
    imag_dir  = base / "imag"
    total_dir = base / "total"
    for d in (real_dir, imag_dir, total_dir):
        d.mkdir(parents=True, exist_ok=True)
    return real_dir, imag_dir, total_dir

def format_latex_subscript(key):
    if "_" in key:
        parts = key.split("_")
        if len(parts) == 2:
            return f"_{{{parts[0]},{parts[1]}}}"
        else:
            return f"_{{{key}}}"
    return f"_{{{key}}}"

def save_excel_data(alphas, results, save_dir):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    all_keys = sorted(list(results.keys()))
    columns = ["Alpha", "Beta"]
    data_cols = [np.asarray(alphas), np.full_like(np.asarray(alphas, dtype=float), FIXED_BETA, dtype=float)]

    for k in all_keys:
        val = np.asarray(results[k])
        columns.extend([f"{k}_Re", f"{k}_Im", f"{k}_Mod"])
        data_cols.extend([np.real(val), np.imag(val), np.abs(val)])


    try:
        import pandas as pd
        df = pd.DataFrame({col: data_cols[i] for i, col in enumerate(columns)})
        outfile = save_path / "SHG_Data_AlphaScan.xlsx"
        df.to_excel(outfile, index=False)
        return
    except Exception as exc:
        pass


    outfile = save_path / "SHG_Data_AlphaScan.csv"
    stacked = np.column_stack(data_cols)
    with open(outfile, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(stacked.tolist())

def _save_lineplot(x, y, xlabel, ylabel, title, fpath, color='blue'):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, linewidth=2.0, color=color)
    ax.set_title(title, fontsize=15)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(fpath, dpi=300)
    plt.close(fig)

def plot_lines_and_save(alphas, results, save_dir=None):
    base_dir = save_dir or OUTDIR
    real_dir, imag_dir, total_dir = _ensure_subdirs(base_dir)


    save_excel_data(alphas, results, base_dir)

    all_keys = list(results.keys())

    for k in all_keys:
        val = results[k]

        val_real_abs = np.abs(np.real(val))
        val_imag_abs = np.abs(np.imag(val))
        val_mod      = np.abs(val)

        if k == "Total":
            label = "yxx"
        else:
            label = format_latex_subscript(k)


        xlabel = r'$\alpha$'


        title_real = rf"$|\mathrm{{Re}}[\chi^{{(2)}}{label}]|$ ($\beta={FIXED_BETA}$)"
        _save_lineplot(alphas, val_real_abs, xlabel, "Magnitude", title_real,
                       real_dir / f"{k}_real_abs.png", color='tab:blue')


        title_imag = rf"$|\mathrm{{Im}}[\chi^{{(2)}}{label}]|$ ($\beta={FIXED_BETA}$)"
        _save_lineplot(alphas, val_imag_abs, xlabel, "Magnitude", title_imag,
                       imag_dir / f"{k}_imag_abs.png", color='tab:orange')


        title_mod = rf"$|\chi^{{(2)}}{label}|$ ($\beta={FIXED_BETA}$)"
        _save_lineplot(alphas, val_mod, xlabel, "Magnitude", title_mod,
                       total_dir / f"{k}_mod.png", color='tab:green')


if __name__ == "__main__":
    if MPI_RANK == 0:
        pass
    alphas, results = compute_lines_alpha_scan()
    if MPI_RANK == 0:
        plot_lines_and_save(alphas, results, save_dir=OUTDIR)
