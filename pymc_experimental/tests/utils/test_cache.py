import os

import pymc as pm

from pymc_experimental.utils.cache import cache_sampling


def test_cache_sampling(tmpdir):

    with pm.Model() as m:
        x = pm.Normal("x", 0, 1)
        y = pm.Normal("y", mu=x, observed=[0, 1, 2])

        cache_prior = cache_sampling(pm.sample_prior_predictive, dir=tmpdir)
        cache_post = cache_sampling(pm.sample, dir=tmpdir)
        cache_pred = cache_sampling(pm.sample_posterior_predictive, dir=tmpdir)
        assert len(os.listdir(tmpdir)) == 0

        prior1, prior2 = (cache_prior(samples=5) for _ in range(2))
        prior3 = cache_sampling(pm.sample_prior_predictive, dir=tmpdir, force_sample=True)(
            samples=5
        )
        assert len(os.listdir(tmpdir)) == 1
        assert prior1.prior["x"].mean() == prior2.prior["x"].mean()
        assert prior2.prior["x"].mean() != prior3.prior["x"].mean()
        assert prior2.prior_predictive["y"].mean() != prior3.prior_predictive["y"].mean()

        post1, post2 = (cache_post(tune=5, draws=5, progressbar=False) for _ in range(2))
        assert len(os.listdir(tmpdir)) == 2
        assert post1.posterior["x"].mean() == post2.posterior["x"].mean()

    # Change model
    with pm.Model() as m:
        x = pm.Normal("x", 0, 1)
        y = pm.Normal("y", mu=x, observed=[0, 1, 2, 3])

        post3 = cache_post(tune=5, draws=5, progressbar=False)
        assert len(os.listdir(tmpdir)) == 3
        assert post3.posterior["x"].mean() != post1.posterior["x"].mean()

        pred1, pred2 = (cache_pred(trace=post3, progressbar=False) for _ in range(2))
        assert len(os.listdir(tmpdir)) == 4
        assert pred1.posterior_predictive["y"].mean() == pred2.posterior_predictive["y"].mean()
        assert "x" not in pred1.posterior_predictive

        # Change kwargs
        pred3 = cache_pred(trace=post3, progressbar=False, var_names=["x"])
        assert len(os.listdir(tmpdir)) == 5
        assert "x" in pred3.posterior_predictive
