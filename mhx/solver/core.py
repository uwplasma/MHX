from __future__ import annotations

import math
import jax.numpy as jnp
import diffrax as dfx

from mhx.solver.plugins import apply_terms, PhysicsTerm

Array = jnp.ndarray


# -----------------------------------------------------------------------------
# Grid & spectral tools
# -----------------------------------------------------------------------------

def estimate_max_dt(
    v_hat: Array,
    B_hat: Array,
    Lx: float,
    Ly: float,
    Lz: float,
    nu: float,
    eta: float,
    CFL_adv: float = 0.4,
    CFL_diff: float = 0.2,
) -> float:
    v = jnp.fft.ifftn(v_hat, axes=(1, 2, 3)).real
    B = jnp.fft.ifftn(B_hat, axes=(1, 2, 3)).real

    v_mag = jnp.sqrt(jnp.sum(v * v, axis=0))
    B_mag = jnp.sqrt(jnp.sum(B * B, axis=0))

    v_char = jnp.max(v_mag + B_mag)

    Nx = v.shape[1]
    Ny = v.shape[2]
    Nz = v.shape[3]
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz
    hmin = jnp.min(jnp.array([dx, dy, dz]))

    dt_adv = jnp.where(v_char > 0.0, CFL_adv * hmin / v_char, 1e9)

    nu_eff = jnp.maximum(nu, eta)
    dt_diff = CFL_diff * hmin * hmin / jnp.maximum(nu_eff, 1e-16)

    dt_max = jnp.minimum(dt_adv, dt_diff)
    return float(dt_max)


def make_grid(Nx: int, Ny: int, Nz: int, Lx: float, Ly: float, Lz: float):
    x = jnp.linspace(0.0, Lx, Nx, endpoint=False)
    y = jnp.linspace(0.0, Ly, Ny, endpoint=False)
    z = jnp.linspace(0.0, Lz, Nz, endpoint=False)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    return X, Y, Z


def make_k_arrays(Nx: int, Ny: int, Nz: int, Lx: float, Ly: float, Lz: float):
    nx = jnp.fft.fftfreq(Nx) * Nx
    ny = jnp.fft.fftfreq(Ny) * Ny
    nz = jnp.fft.fftfreq(Nz) * Nz
    NX, NY, NZ = jnp.meshgrid(nx, ny, nz, indexing="ij")

    kx = 2.0 * jnp.pi * NX / Lx
    ky = 2.0 * jnp.pi * NY / Ly
    kz = 2.0 * jnp.pi * NZ / Lz

    k2 = kx**2 + ky**2 + kz**2
    k2 = jnp.where(k2 == 0.0, 1.0, k2)
    return kx, ky, kz, k2, NX, NY, NZ


def make_dealias_mask(Nx: int, Ny: int, Nz: int, NX: Array, NY: Array, NZ: Array):
    kx_cut = Nx // 3
    ky_cut = Ny // 3
    kz_cut = Nz // 3

    mask = (jnp.abs(NX) <= kx_cut) & (jnp.abs(NY) <= ky_cut) & (jnp.abs(NZ) <= kz_cut)
    return mask


def project_div_free(F_hat: Array, kx: Array, ky: Array, kz: Array, k2: Array) -> Array:
    k_dot_F = kx * F_hat[0] + ky * F_hat[1] + kz * F_hat[2]
    F_hat_proj = jnp.stack(
        [
            F_hat[0] - kx * k_dot_F / k2,
            F_hat[1] - ky * k_dot_F / k2,
            F_hat[2] - kz * k_dot_F / k2,
        ],
        axis=0,
    )
    return F_hat_proj


def grad_from_hat(f_hat: Array, kx: Array, ky: Array, kz: Array) -> Array:
    df_dx_hat = 1j * kx * f_hat
    df_dy_hat = 1j * ky * f_hat
    df_dz_hat = 1j * kz * f_hat

    df_dx = jnp.fft.ifftn(df_dx_hat, axes=(0, 1, 2)).real
    df_dy = jnp.fft.ifftn(df_dy_hat, axes=(0, 1, 2)).real
    df_dz = jnp.fft.ifftn(df_dz_hat, axes=(0, 1, 2)).real

    return jnp.stack([df_dx, df_dy, df_dz], axis=0)


def grad_vec_from_hat(F_hat: Array, kx: Array, ky: Array, kz: Array) -> Array:
    df_dx = grad_from_hat(F_hat[0], kx, ky, kz)
    df_dy = grad_from_hat(F_hat[1], kx, ky, kz)
    df_dz = grad_from_hat(F_hat[2], kx, ky, kz)

    grad_F = jnp.stack(
        [
            jnp.stack([df_dx[0], df_dy[0], df_dz[0]], axis=0),
            jnp.stack([df_dx[1], df_dy[1], df_dz[1]], axis=0),
            jnp.stack([df_dx[2], df_dy[2], df_dz[2]], axis=0),
        ],
        axis=0,
    )
    return grad_F


def directional_derivative_vec(A: Array, grad_B: Array) -> Array:
    return jnp.einsum("i...,ij...->j...", A, grad_B)


def curl_from_hat(B_hat: Array, kx: Array, ky: Array, kz: Array) -> Array:
    Bx_hat, By_hat, Bz_hat = B_hat[0], B_hat[1], B_hat[2]

    Jx_hat = 1j * (ky * Bz_hat - kz * By_hat)
    Jy_hat = 1j * (kz * Bx_hat - kx * Bz_hat)
    Jz_hat = 1j * (kx * By_hat - ky * Bx_hat)

    J_hat = jnp.stack([Jx_hat, Jy_hat, Jz_hat], axis=0)
    J = jnp.fft.ifftn(J_hat, axes=(1, 2, 3)).real
    return J


# -----------------------------------------------------------------------------
# Energies & dissipation
# -----------------------------------------------------------------------------

def energy_from_hat(v_hat: Array, B_hat: Array, Lx: float, Ly: float, Lz: float):
    v = jnp.fft.ifftn(v_hat, axes=(1, 2, 3)).real
    B = jnp.fft.ifftn(B_hat, axes=(1, 2, 3)).real
    dv = (Lx * Ly * Lz) / (v[0].size)

    v2 = jnp.sum(v * v, axis=0)
    B2 = jnp.sum(B * B, axis=0)
    E_kin = 0.5 * jnp.sum(v2) * dv
    E_mag = 0.5 * jnp.sum(B2) * dv
    return E_kin, E_mag


def dissipation_rates(v_hat: Array, B_hat: Array, k2: Array, nu: float, eta: float, Lx: float, Ly: float, Lz: float):
    v_power = jnp.sum(jnp.abs(v_hat) ** 2, axis=0)
    B_power = jnp.sum(jnp.abs(B_hat) ** 2, axis=0)

    volume = Lx * Ly * Lz
    Npoints = v_hat.shape[1] * v_hat.shape[2] * v_hat.shape[3]
    factor = volume / (Npoints**2)

    eps_visc = nu * factor * jnp.sum(k2 * v_power)
    eps_ohm = eta * factor * jnp.sum(k2 * B_power)
    return eps_visc, eps_ohm


# -----------------------------------------------------------------------------
# RHS builders
# -----------------------------------------------------------------------------

def make_mhd_rhs_original(nu: float, eta: float, kx: Array, ky: Array, kz: Array, k2: Array, mask_dealias: Array, terms: list[PhysicsTerm] | None = None):
    def rhs(t, y_hat, args_unused):
        v_hat, B_hat = y_hat

        v_hat_ = v_hat * mask_dealias
        B_hat_ = B_hat * mask_dealias

        v_hat_ = project_div_free(v_hat_, kx, ky, kz, k2)
        B_hat_ = project_div_free(B_hat_, kx, ky, kz, k2)

        v = jnp.fft.ifftn(v_hat_, axes=(1, 2, 3)).real
        B = jnp.fft.ifftn(B_hat_, axes=(1, 2, 3)).real

        grad_v = grad_vec_from_hat(v_hat_, kx, ky, kz)
        grad_B = grad_vec_from_hat(B_hat_, kx, ky, kz)

        v_dot_grad_v = directional_derivative_vec(v, grad_v)
        B_dot_grad_B = directional_derivative_vec(B, grad_B)

        adv_v = v_dot_grad_v - B_dot_grad_B

        curl_B = curl_from_hat(B_hat_, kx, ky, kz)
        JxB = jnp.cross(curl_B.transpose(1, 2, 3, 0), B.transpose(1, 2, 3, 0))
        JxB = JxB.transpose(3, 0, 1, 2)

        dv_dt = -adv_v + JxB

        v_dot_grad_B = directional_derivative_vec(v, grad_B)
        B_dot_grad_v = directional_derivative_vec(B, grad_v)
        dB_dt = B_dot_grad_v - v_dot_grad_B

        dv_hat = jnp.fft.fftn(dv_dt, axes=(1, 2, 3))
        dB_hat = jnp.fft.fftn(dB_dt, axes=(1, 2, 3))

        dv_hat = project_div_free(dv_hat, kx, ky, kz, k2)
        dB_hat = project_div_free(dB_hat, kx, ky, kz, k2)

        dv_hat = dv_hat - nu * k2 * v_hat_
        dB_hat = dB_hat - eta * k2 * B_hat_

        if terms:
            dv_add, dB_add = apply_terms(terms, t=t, v_hat=v_hat_, B_hat=B_hat_, kx=kx, ky=ky, kz=kz, k2=k2, mask_dealias=mask_dealias)
            dv_hat = dv_hat + dv_add
            dB_hat = dB_hat + dB_add

        return (dv_hat, dB_hat)

    return rhs


def make_mhd_rhs_forcefree(nu: float, eta: float, kx: Array, ky: Array, kz: Array, k2: Array, mask_dealias: Array, B0_hat_const: Array, Nv0_hat_const: Array, terms: list[PhysicsTerm] | None = None):
    def rhs(t, y_hat, args_unused):
        v_hat, B_hat = y_hat

        v_hat_ = v_hat * mask_dealias
        B_hat_ = B_hat * mask_dealias

        v_hat_ = project_div_free(v_hat_, kx, ky, kz, k2)
        B_hat_ = project_div_free(B_hat_, kx, ky, kz, k2)

        v = jnp.fft.ifftn(v_hat_, axes=(1, 2, 3)).real
        B = jnp.fft.ifftn(B_hat_, axes=(1, 2, 3)).real

        grad_v = grad_vec_from_hat(v_hat_, kx, ky, kz)
        grad_B = grad_vec_from_hat(B_hat_, kx, ky, kz)

        v_dot_grad_v = directional_derivative_vec(v, grad_v)
        B_dot_grad_B = directional_derivative_vec(B, grad_B)

        adv_v = v_dot_grad_v - B_dot_grad_B

        curl_B = curl_from_hat(B_hat_, kx, ky, kz)
        JxB = jnp.cross(curl_B.transpose(1, 2, 3, 0), B.transpose(1, 2, 3, 0))
        JxB = JxB.transpose(3, 0, 1, 2)

        dv_dt = -adv_v + JxB

        v_dot_grad_B = directional_derivative_vec(v, grad_B)
        B_dot_grad_v = directional_derivative_vec(B, grad_v)
        dB_dt = B_dot_grad_v - v_dot_grad_B

        dv_hat = jnp.fft.fftn(dv_dt, axes=(1, 2, 3))
        dB_hat = jnp.fft.fftn(dB_dt, axes=(1, 2, 3))

        dv_hat = project_div_free(dv_hat, kx, ky, kz, k2)
        dB_hat = project_div_free(dB_hat, kx, ky, kz, k2)

        dv_hat = dv_hat - nu * k2 * v_hat_
        dB_hat = dB_hat - eta * k2 * B_hat_

        # subtract equilibrium force
        dv_hat = dv_hat - Nv0_hat_const
        dB_hat = dB_hat - eta * k2 * B0_hat_const

        if terms:
            dv_add, dB_add = apply_terms(terms, t=t, v_hat=v_hat_, B_hat=B_hat_, kx=kx, ky=ky, kz=kz, k2=k2, mask_dealias=mask_dealias)
            dv_hat = dv_hat + dv_add
            dB_hat = dB_hat + dB_add

        return (dv_hat, dB_hat)

    return rhs


# -----------------------------------------------------------------------------
# Time integration
# -----------------------------------------------------------------------------

def run_time_integration(
    rhs,
    *,
    t0: float,
    t1: float,
    dt0: float,
    n_frames: int,
    y0,
):
    term = dfx.ODETerm(rhs)
    solver = dfx.Dopri8()
    stepsize_controller = dfx.PIDController(rtol=1e-7, atol=1e-7)
    ts_save = jnp.linspace(t0, t1, n_frames)
    saveat = dfx.SaveAt(ts=ts_save)

    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=None,
        saveat=saveat,
        max_steps=int((t1 - t0) / dt0) + 10_000,
        stepsize_controller=stepsize_controller,
        progress_meter=dfx.TqdmProgressMeter(),
    )
    return sol


# -----------------------------------------------------------------------------
# Theory helpers (kept for compatibility)
# -----------------------------------------------------------------------------

def fkr_gamma(B0: float, a: float, Ly: float, eta: float):
    k = 2.0 * math.pi / Ly
    delta_prime_a = 2.0 * (1.0 / (k * a) - k * a)
    S_a = (a * B0) / eta
    gamma = 0.55 * (S_a ** (-3.0 / 5.0)) * (delta_prime_a ** (4.0 / 5.0))
    return gamma, S_a, delta_prime_a


def compute_k_arrays_np(Nx: int, Ny: int, Nz: int, Lx: float, Ly: float, Lz: float):
    nx = jnp.fft.fftfreq(Nx) * Nx
    ny = jnp.fft.fftfreq(Ny) * Ny
    nz = jnp.fft.fftfreq(Nz) * Nz
    NX, NY, NZ = jnp.meshgrid(nx, ny, nz, indexing="ij")

    kx = 2.0 * jnp.pi * NX / Lx
    ky = 2.0 * jnp.pi * NY / Ly
    kz = 2.0 * jnp.pi * NZ / Lz

    k2 = kx**2 + ky**2 + kz**2
    k2 = jnp.where(k2 == 0.0, 1.0, k2)
    return kx, ky, kz, k2, NX, NY, NZ


def energy_from_hat_np(v_hat: Array, B_hat: Array, Lx: float, Ly: float, Lz: float):
    v = jnp.fft.ifftn(v_hat, axes=(1, 2, 3)).real
    B = jnp.fft.ifftn(B_hat, axes=(1, 2, 3)).real
    dv = (Lx * Ly * Lz) / (v[0].size)

    v2 = jnp.sum(v * v, axis=0)
    B2 = jnp.sum(B * B, axis=0)
    E_kin = 0.5 * jnp.sum(v2) * dv
    E_mag = 0.5 * jnp.sum(B2) * dv
    return E_kin, E_mag
