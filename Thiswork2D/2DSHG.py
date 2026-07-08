import os
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


Na, Nb   = 100, 100
Nkx, Nky = 1400, 1400
ALPHA_MIN, ALPHA_MAX = 0.0, 2.0
BETA_MIN,  BETA_MAX  = 0.0, 1.0
KX_MIN, KX_MAX = -np.pi, np.pi
KY_MIN, KY_MAX =  0.0, 2.0*np.pi


OMEGA      = 5

ETA_PV1 = 0.01
ETA_PV2 = 0.01


OUTDIR    = "SHG_Output_MPI"


USE_WILSON_LOOP = True


FONT_TITLE = 17
FONT_LABEL = 19
FONT_TICK_AXIS = 19
FONT_TICK_CBAR = 12


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


def kernel_plus(x, eta):
    return 1.0 / (x + 1j*eta/2.0)

def kernel_minus(x, eta):
    return 1.0 / (x - 1j*eta/2.0)


def eigh_2x2_herm_batch(H):
    return np.linalg.eigh(H)

def mat_elem(U_left, Op, U_right):
    left  = np.conj(U_left)[..., None, :]
    right = U_right[..., :, None]
    return (left @ Op @ right)[..., 0, 0]

def band_vectors(U, band_idx):
    return U[..., :, band_idx]

def overlap(psi, phi):
    return np.sum(np.conj(psi) * phi, axis=-1)


def roll_plus(arr, axis):
    return np.roll(arr, -1, axis=axis)

def roll_minus(arr, axis):
    return np.roll(arr, +1, axis=axis)

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

    local_nkx = kx.shape[0]
    dkx = (KX_MAX - KX_MIN) / local_nkx
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


def _compute_one(ia, ib, a, b, KX, KY, d_kx, d_ky):
    terms = subterms_on_kgrid_numeric(KX, KY, a, b)
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

    return (ia, ib,
            I1a_1, I1a_2, I1b_1, I1b_2,
            I2a_1, I2a_2, I2b_1, I2b_2, I2c_1, I2c_2)


def compute_maps_mpi():
    alphas = np.linspace(ALPHA_MIN, ALPHA_MAX, Na)
    betas  = np.linspace(BETA_MIN,  BETA_MAX,  Nb)
    kx = np.linspace(KX_MIN, KX_MAX, Nkx, endpoint=False)
    ky = np.linspace(KY_MIN, KY_MAX, Nky, endpoint=False)
    d_kx = (KX_MAX - KX_MIN) / Nkx
    d_ky = (KY_MAX - KY_MIN) / Nky
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    keys_sub = ["T1a_1", "T1a_2", "T1b_1", "T1b_2",
                "T2a_1", "T2a_2", "T2b_1", "T2b_2", "T2c_1", "T2c_2"]
    keys_main = ["T1a", "T1b", "T2a", "T2b", "T2c"]

    full_tasks = [(ia, ib, a, b) for ia, a in enumerate(alphas) for ib, b in enumerate(betas)]
    my_tasks = full_tasks[rank::size]

    local_results = []
    iterator = my_tasks
    for ia, ib, a, b in iterator:
        local_results.append(_compute_one(ia, ib, a, b, KX, KY, d_kx, d_ky))

    all_results_list = comm.gather(local_results, root=0)

    if rank != 0:
        return alphas, betas, None

    maps = {k: np.zeros((Nb, Na), dtype=complex) for k in keys_sub + keys_main + ["Total"]}

    flat_results = []
    for res_list in all_results_list:
        flat_results.extend(res_list)

    for res in flat_results:
        (ia, ib,
         i1a1, i1a2, i1b1, i1b2,
         i2a1, i2a2, i2b1, i2b2, i2c1, i2c2) = res

        maps["T1a_1"][ib, ia] = i1a1
        maps["T1a_2"][ib, ia] = i1a2
        maps["T1b_1"][ib, ia] = i1b1
        maps["T1b_2"][ib, ia] = i1b2
        maps["T2a_1"][ib, ia] = i2a1
        maps["T2a_2"][ib, ia] = i2a2
        maps["T2b_1"][ib, ia] = i2b1
        maps["T2b_2"][ib, ia] = i2b2
        maps["T2c_1"][ib, ia] = i2c1
        maps["T2c_2"][ib, ia] = i2c2

        v_t1a = i1a1 + i1a2
        v_t1b = i1b1 + i1b2
        v_t2a = i2a1 + i2a2
        v_t2b = i2b1 + i2b2
        v_t2c = i2c1 + i2c2

        maps["T1a"][ib, ia] = v_t1a
        maps["T1b"][ib, ia] = v_t1b
        maps["T2a"][ib, ia] = v_t2a
        maps["T2b"][ib, ia] = v_t2b
        maps["T2c"][ib, ia] = v_t2c
        maps["Total"][ib, ia] = v_t1a + v_t1b + v_t2a + v_t2b + v_t2c

    return alphas, betas, maps


def _ensure_subdirs(base_dir):
    base = Path(base_dir or ".")
    real_dir  = base / "real"
    imag_dir  = base / "imag"
    total_dir = base / "total"
    for d in (real_dir, imag_dir, total_dir):
        d.mkdir(parents=True, exist_ok=True)
    return real_dir, imag_dir, total_dir

def _save_heatmap(x_vec, y_vec, data, title, fpath):
    fig = plt.figure(figsize=(6.5, 5.2))
    ax = fig.add_subplot(111)
    im = ax.pcolormesh(x_vec, y_vec, data, shading='auto', cmap='viridis')
    ax.set_title(title, fontsize=FONT_TITLE)
    ax.set_xlabel(r'$\alpha$', fontsize=FONT_LABEL)
    ax.set_ylabel(r'$\beta$', fontsize=FONT_LABEL)
    ax.tick_params(axis='both', labelsize=FONT_TICK_AXIS)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=FONT_TICK_CBAR)
    fig.tight_layout()
    fig.savefig(fpath, dpi=300)
    plt.close(fig)


def format_latex_subscript(key):
    prefix = "yxx"
    if key == "Total":
        return f"_{{{prefix}}}"
    if "_" in key:
        parts = key.split("_")
        return f"_{{{prefix}, {parts[0]}_{parts[1]}}}"
    return f"_{{{prefix}, {key}}}"

def plot_all(alphas, betas, maps, save_dir=None):
    keys_sub = ["T1a_1", "T1a_2", "T1b_1", "T1b_2",
                "T2a_1", "T2a_2", "T2b_1", "T2b_2", "T2c_1", "T2c_2"]
    keys_main = ["T1a", "T1b", "T2a", "T2b", "T2c"]
    all_keys = keys_sub + keys_main + ["Total"]

    real_dir, imag_dir, total_dir = _ensure_subdirs(save_dir or OUTDIR)

    for k in all_keys:
        val = maps[k]
        val_real_abs = np.abs(np.real(val))
        val_imag_abs = np.abs(np.imag(val))
        val_mod      = np.abs(val)

        label = format_latex_subscript(k)

        title_real = rf"$|\mathrm{{Re}}[\chi^{{(2)}}{label}]|$"
        _save_heatmap(alphas, betas, val_real_abs, title_real, real_dir / f"{k}_real.png")

        title_imag = rf"$|\mathrm{{Im}}[\chi^{{(2)}}{label}]|$"
        _save_heatmap(alphas, betas, val_imag_abs, title_imag, imag_dir / f"{k}_imag.png")

        title_mod = rf"$|\chi^{{(2)}}{label}|$"
        _save_heatmap(alphas, betas, val_mod, title_mod, total_dir / f"{k}_mod.png")


if __name__ == "__main__":
    if rank == 0:
        os.makedirs(OUTDIR, exist_ok=True)
    comm.Barrier()

    alphas, betas, maps = compute_maps_mpi()

    if rank == 0:
        plot_all(alphas, betas, maps, save_dir=OUTDIR)
