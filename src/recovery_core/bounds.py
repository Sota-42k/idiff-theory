# Extracted from identifiability-research-1, src/idiff/bounds.py at commit
# b4cd012dc31a5305227ddd2c4f385e88fc651af3 (sha256 8dadd6237d0a5a53fbcc5a62ba0181773491f9ceec30c7bf79120cbaff0d15d9).
# Modified in this repository: import path idiff.data -> recovery_core.generator. No numerical change.
# No upstream TCL ancestor; this is a project addition, not a TCL-derived file.
# See docs/provenance.md.
"""Conservative comparison constants and empirical E3 right-hand sides.

For y=artanh(z) with uniform independent z, each coordinate has density
sech(y)^2/2. Its density divided by the smaller tail probability is >=1,
so the one-dimensional Cheeger inequality gives C_P<=4. Tensorization keeps
4 in any dimension. If D is the largest oscillation of a conditional latent
log-density's exponent on the box, each normalized density relative to uniform
lies between exp(-D) and exp(D), as does their mixture. Bounded-density
comparison thus gives C_P<=4 exp(2D). Mapping by h multiplies this by at most
Lip(h)^2 <= 1.1^6. Similarly p0/p_u <= exp(2D) for every u.

These deliberately loose analytic comparison formulas avoid pretending that a
sampled maximum or a numerical spectral gap is a certified bound. Floating
point evaluation is not interval certification, and finite-sample field errors
and approximate potentials still make the reported panel an empirical
population-bound diagnostic, not a theorem certificate.
"""

import numpy as np

from recovery_core.generator import Params


def comparison_constants(params: Params, *, layers: int = 3) -> dict[str, float | str]:
    a, b = params.a, params.b
    vertex = np.clip(-a/(2*b), -1, 1)
    values = np.stack((-a+b, a+b, a*vertex+b*vertex**2))
    oscillations = (values.max(axis=0)-values.min(axis=0)).sum(axis=1)
    D = float(oscillations.max())
    M = float(np.exp(2*D))
    return {"C_P_comparison": 4*M*1.1**(2*layers), "M_comparison": M,
            "exponent_oscillation": D, "constant_method": "bounded_density_comparison",
            "certificate_status": "analytic formula evaluated in float64; empirical errors"}


def bound_rhs(cp: float, sigma_L: float, sigma_C: float, M: float,
              J_ref_sum: float, J_phi: float, J_individual: np.ndarray) -> dict[str, float]:
    if min(cp, sigma_L, sigma_C, M) <= 0:
        raise ValueError("Constants and smallest singular values must be positive")
    q = len(J_individual)-1
    return {
        "B_ref": cp/sigma_L**2*J_ref_sum,
        "B_ref_bridge": 2*cp*M/sigma_L**2*float(J_individual[1:].sum()+q*J_individual[0]),
        "B_phi": cp/sigma_C**2*J_phi,
        "B_phi_bridge": cp*M/sigma_C**2*float(J_individual.mean()),
        "B_phi_raw": cp*J_phi,
    }
