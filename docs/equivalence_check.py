"""One-off equivalence check: extracted recovery_core vs the immediate source idiff.

Not shipped as a test: the new repository must not depend on the original checkout.
"""
import argparse
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-src", type=Path, required=True,
                    help="src directory of the historical immediate-source checkout")
args = parser.parse_args()
if not (args.source_src / "idiff").is_dir():
    parser.error("--source-src must contain the historical idiff package")
sys.path.insert(0, str(args.source_src.resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from idiff import data as d_old, readout as r_old, metrics as m_old, directions as dir_old, bounds as b_old
from recovery_core import generator as d_new, readout as r_new, metrics as m_new, directions as dir_new, bounds as b_new

fails = []
def same(name, a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = a.shape == b.shape and np.array_equal(a, b)
    print(f"{'OK ' if ok else 'BAD'}  {name}  max|diff|={0.0 if ok else float(np.max(np.abs(a-b)))}")
    if not ok: fails.append(name)

for n in (2, 3):
    for identity in (True, False):
        po, pn = d_old.draw_parameters(n, 41, 1.), d_new.draw_parameters(n, 41, 1.)
        same(f"natural n={n}", po.natural, pn.natural)
        same(f"L n={n}", po.L, pn.L); same(f"kappa n={n}", po.kappa, pn.kappa)
        same(f"log_z n={n}", po.log_z, pn.log_z); same(f"cdf n={n}", po.cdf, pn.cdf)
        mo = d_old.mixing(n, 22, identity=identity); mn = d_new.mixing(n, 22, identity=identity)
        same(f"rotations n={n} id={identity}", mo.rotations, mn.rotations)
        z = np.random.default_rng(5).uniform(-.8, .8, (64, n))
        x = mo.g(z)
        same(f"g n={n} id={identity}", x, mn.g(z))
        same(f"g_inv n={n} id={identity}", mo.g_inv(x), mn.g_inv(x))
        same(f"jac n={n} id={identity}", mo.jac(z), mn.jac(z))
        same(f"jac_inv n={n} id={identity}", mo.jac_inv(x), mn.jac_inv(x))
        same(f"logdet n={n} id={identity}", mo.logdet_jac_inv(x), mn.logdet_jac_inv(x))
        same(f"grad_logdet n={n} id={identity}", mo.grad_logdet_jac_inv(x), mn.grad_logdet_jac_inv(x))
        oo, on = d_old.Oracle(po, mo), d_new.Oracle(pn, mn)
        same(f"scores n={n} id={identity}", oo.scores(x), on.scores(x))
        same(f"readout n={n} id={identity}", oo.readout(x), on.readout(x))
        same(f"contrast n={n} id={identity}", oo.contrast(x), on.contrast(x))
        same(f"statistic_jac n={n} id={identity}", oo.statistic_jacobian(x), on.statistic_jacobian(x))
        same(f"log_density n={n} id={identity}", oo.log_density(x), on.log_density(x))
        same(f"overlap n={n} id={identity}", oo.overlap_samples(x), on.overlap_samples(x))
        same(f"sample_latents n={n}", d_old.sample_latents(po, 1, 500, np.random.default_rng(3)),
             d_new.sample_latents(pn, 1, 500, np.random.default_rng(3)))
        # readout / metrics / directions / bounds on the same fixture
        fo, fn = r_old.Features.categorical(po.q), r_new.Features.categorical(pn.q)
        same(f"xi n={n}", fo.xi, fn.xi)
        same(f"C_phi n={n}", fo.oracle_alignment(po.lambdas), fn.oracle_alignment(pn.lambdas))
        s = oo.scores(x)
        same(f"reference_field n={n} id={identity}", r_old.reference_field(s), r_new.reference_field(s))
        same(f"regression_field n={n} id={identity}", r_old.regression_field(s, fo), r_new.regression_field(s, fn))
        pot_o = r_old.fit_potential(x, r_old.reference_field(s), n_features=32, seed=9)
        pot_n = r_new.fit_potential(x, r_new.reference_field(s), n_features=32, seed=9)
        same(f"potential n={n} id={identity}", pot_o(x), pot_n(x))
        same(f"potential_grad n={n} id={identity}", pot_o.gradient(x), pot_n.gradient(x))
        cst_o, cst_n = b_old.comparison_constants(po), b_new.comparison_constants(pn)
        same(f"C_P n={n}", cst_o["C_P_comparison"], cst_n["C_P_comparison"])
        same(f"M n={n}", cst_o["M_comparison"], cst_n["M_comparison"])

rng = np.random.default_rng(4)
cal, test = rng.normal(size=(100, 3)), rng.normal(size=(70, 3))
A = np.diag([2., -1., 3.])
fo_, fn_ = m_old.fit_affine(cal, cal @ A+2), m_new.fit_affine(cal, cal @ A+2)
same("affine A", fo_.A, fn_.A); same("affine b", fo_.b, fn_.b)
same("affine_mse", m_old.affine_residual(fo_, test, test @ A+3)["affine_mse"],
     m_new.affine_residual(fn_, test, test @ A+3)["affine_mse"])
same("oracle_align", m_old.oracle_alignment_error(test, A), m_new.oracle_alignment_error(test, A))
G = np.diag([4., 2., .5]) @ np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
Uo, so, Vo = dir_old.singular_directions(G); Un, sn, Vn = dir_new.singular_directions(G)
same("svd U", Uo, Un); same("svd s", so, sn); same("svd V", Vo, Vn)

print("\nFAILURES:", fails if fails else "none")
if fails:
    raise SystemExit(1)
