import numpy as np
import pymc as pm
import pytensor.tensor as pt

from pymc_experimental.gp.pytensor_gp import GP, ExpQuad


def test_exp_quad():
    x = pt.arange(3)[:, None]
    ls = pt.ones(())
    cov = ExpQuad(x, ls=ls).eval()
    expected_distance = np.array([[0.0, 1.0, 4.0], [1.0, 0.0, 1.0], [4.0, 1.0, 0.0]])

    np.testing.assert_allclose(cov, np.exp(-0.5 * expected_distance))


# @pytest.fixture(scope="session")
def latent_model():
    with pm.Model() as m:
        X = pm.Data("X", np.arange(3)[:, None])
        y = np.full(3, np.pi)
        ls = 1.0
        cov = ExpQuad(X, ls=ls)
        gp = GP("gp", cov=cov)

        sigma = 1.0
        obs = pm.Normal("obs", mu=gp, sigma=sigma, observed=y)

    return m


def latent_model_old_API():
    with pm.Model() as m:
        X = pm.Data("X", np.arange(3)[:, None])
        y = np.full(3, np.pi)
        ls = 1.0
        cov = pm.gp.cov.ExpQuad(1, ls)
        gp_class = pm.gp.Latent(cov_func=cov)
        gp = gp_class.prior("gp", X, reparameterize=False)

        sigma = 1.0
        obs = pm.Normal("obs", mu=gp, sigma=sigma, observed=y)

    return m, gp_class


def test_latent_model_prior():
    m = latent_model()
    ref_m, _ = latent_model_old_API()

    prior = pm.draw(m["gp"], draws=1000)
    prior_ref = pm.draw(ref_m["gp"], draws=1000)

    np.testing.assert_allclose(
        prior.mean(),
        prior_ref.mean(),
        atol=0.1,
    )

    np.testing.assert_allclose(
        prior.std(),
        prior_ref.std(),
        rtol=0.1,
    )


def test_latent_model_logp():
    m = latent_model()
    ip = m.initial_point()

    ref_m, _ = latent_model_old_API()

    np.testing.assert_allclose(
        m.compile_logp()(ip),
        ref_m.compile_logp()(ip),
        rtol=1e-6,
    )


import arviz as az


def gp_conditional(model, gp, Xnew, jitter=1e-6):
    def _build_conditional(self, Xnew, f, cov, jitter):
        X, ls = cov.owner.inputs

        Kxx = cov
        Kxs = cov.owner.op.build_covariance(X, Xnew, ls=ls)
        Kss = cov.owner.op.build_covariance(Xnew, ls=ls)

        L = pt.linalg.cholesky(Kxx + pt.eye(X.shape[0]) * jitter)
        # TODO: Use cho_solve
        A = pt.linalg.solve_triangular(L, Kxs, lower=True)
        v = pt.linalg.solve_triangular(L, f, lower=True)

        mu = (A.mT @ v).T  # Vector?
        cov = Kss - (A.mT @ A)

        return mu, cov

    with model.copy() as new_m:
        gp = new_m[gp.name]
        _, cov = gp.owner.op.dist_params(gp.owner)
        mu_star, cov_star = _build_conditional(None, Xnew, gp, cov, jitter)
        gp_star = pm.MvNormal("gp_star", mu_star, cov_star)
        return new_m


def test_latent_model_predict_new_x():
    rng = np.random.default_rng(0)
    new_x = np.array([3, 4])[:, None]

    m = latent_model()
    ref_m, ref_gp_class = latent_model_old_API()

    posterior_idata = az.from_dict({"gp": rng.normal(np.pi, 1e-3, size=(4, 1000, 2))})

    # with gp_extend_to_new_x(m):
    with gp_conditional(m, m["gp"], new_x):
        pred = (
            pm.sample_posterior_predictive(posterior_idata, var_names=["gp_star"])
            .posterior_predictiev["gp"]
            .values
        )

    with ref_m:
        gp_star = ref_gp_class.conditional("gp_star", Xnew=new_x)
        pred_ref = (
            pm.sample_posterior_predictive(posterior_idata, var_names=["gp_star"])
            .posterior_predictive["gp"]
            .values
        )

    np.testing.assert_allclose(
        pred.mean(),
        pred_ref.mean(),
        atol=0.1,
    )

    np.testing.assert_allclose(
        pred.std(),
        pred_ref.std(),
        rtol=0.1,
    )


#
# def test_marginal_sigma_rewrites_to_white_noise_cov(marginal_model, ):
#     obs = marginal_model["obs"]
#
#     # TODO: Bring these checks back after we implement marginalization of the GP RV
#     #
#     # assert sum(isinstance(var.owner.op, pm.Normal.rv_type)
#     #            for var in ancestors([obs])
#     #            if var.owner is not None) == 1
#     #
#     f = pm.compile_pymc([], obs)
#     #
#     # assert not any(isinstance(node.op, pm.Normal.rv_type) for node in f.maker.fgraph.apply_nodes)
#
#     draws = np.stack([f() for _ in range(10_000)])
#     empirical_cov = np.cov(draws.T)
#
#     expected_distance = np.array([[0.0, 1.0, 4.0], [1.0, 0.0, 1.0], [4.0, 1.0, 0.0]])
#
#     np.testing.assert_allclose(
#         empirical_cov, np.exp(-0.5 * expected_distance) + np.eye(3), atol=0.1, rtol=0.1
#     )
#
#
# def test_marginal_gp_logp(marginal_model):
#     expected_logps = {"obs": -8.8778}
#     point_logps = marginal_model.point_logps(round_vals=4)
#     for v1, v2 in zip(point_logps.values(), expected_logps.values()):
#         np.testing.assert_allclose(v1, v2, atol=1e-6)
