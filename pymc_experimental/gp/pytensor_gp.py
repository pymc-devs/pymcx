from collections.abc import Sequence

import pymc as pm
import pytensor.tensor as pt

from pymc.distributions.distribution import Continuous
from pymc.model.fgraph import fgraph_from_model, model_free_rv, model_from_fgraph
from pytensor import Variable
from pytensor.compile.builders import OpFromGraph


class GPCovariance(OpFromGraph):
    """OFG representing a GP covariance"""

    @staticmethod
    def square_dist_Xs(X, Xs, ls):
        assert X.ndim == 2, "Complain to Bill about it"
        assert Xs.ndim == 2, "Complain to Bill about it"

        X = X / ls
        Xs = Xs / ls

        X2 = pt.sum(pt.square(X), axis=-1)
        Xs2 = pt.sum(pt.square(Xs), axis=-1)

        sqd = -2.0 * X @ Xs.mT + (X2[..., :, None] + Xs2[..., None, :])
        # sqd = -2.0 * pt.dot(X, pt.transpose(Xs)) + (
        #         pt.reshape(X2, (-1, 1)) + pt.reshape(Xs2, (1, -1))
        # )

        return pt.clip(sqd, 0, pt.inf)

    @staticmethod
    def square_dist(X, ls):
        X = X / ls
        X2 = pt.sum(pt.square(X), axis=-1)
        sqd = -2.0 * X @ X.mT + (X2[..., :, None] + X2[..., None, :])

        return sqd


class ExpQuadCov(GPCovariance):
    """
    ExpQuad covariance function
    """

    @classmethod
    def exp_quad_full(cls, X, Xs, ls):
        return pt.exp(-0.5 * cls.square_dist_Xs(X, Xs, ls))

    @classmethod
    def build_covariance(cls, X, Xs=None, *, ls):
        X = pt.as_tensor(X)
        if Xs is None:
            Xs = X
        else:
            Xs = pt.as_tensor(Xs)
        ls = pt.as_tensor(ls)

        out = cls.exp_quad_full(X, Xs, ls)
        if Xs is X:
            return cls(inputs=[X, ls], outputs=[out])(X, ls)
        else:
            return cls(inputs=[X, Xs, ls], outputs=[out])(X, Xs, ls)


def ExpQuad(X, X_new=None, *, ls):
    return ExpQuadCov.build_covariance(X, X_new, ls=ls)


# class WhiteNoiseCov(GPCovariance):
#     @classmethod
#     def white_noise_full(cls, X, sigma):
#         X_shape = tuple(X.shape)
#         shape = X_shape[:-1] + (X_shape[-2],)
#
#         return _delta(shape, normalize_axis_tuple((-1, -2), X.ndim)) * sigma**2
#
#     @classmethod
#     def build_covariance(cls, X, sigma):
#         X = pt.as_tensor(X)
#         sigma = pt.as_tensor(sigma)
#
#         ofg = cls(inputs=[X, sigma], outputs=[cls.white_noise_full(X, sigma)])
#         return ofg(X, sigma)

#
# def WhiteNoise(X, sigma):
#     return WhiteNoiseCov.build_covariance(X, sigma)
#


class GP_RV(pm.MvNormal.rv_type):
    name = "gaussian_process"
    signature = "(n),(n,n)->(n)"
    dtype = "floatX"
    _print_name = ("GP", "\\operatorname{GP}")


class GP(Continuous):
    rv_type = GP_RV
    rv_op = GP_RV()

    @classmethod
    def dist(cls, cov, **kwargs):
        # return Assert(msg="Don't know what a GP_RV is")(False)
        cov = pt.as_tensor(cov)
        mu = pt.zeros(cov.shape[-1])
        return super().dist([mu, cov], **kwargs)


def conditional_gp(
    model,
    gp: Variable | str,
    Xnew,
    *,
    jitter=1e-6,
    dims: Sequence[str] = (),
    inline: bool = False,
):
    """
    Condition a GP on new data.

    Parameters
    ----------
    model: Model
    gp: Variable | str
        The GP to condition on.
    Xnew: Tensor-like
        New data to condition the GP on.
    jitter: float, default=1e-6
        Jitter to add to the new GP covariance matrix.
    dims: Sequence[str], default=()
        Dimensions of the new GP.
    inline: bool, default=False
        Whether to inline the new GP in place of the old one. This is not always a safe operation.
        If True, any variables that depend on the GP will be updated to depend on the new GP.

    Returns
    -------
    Conditional model: Model
        A new model with a GP free RV named f"{gp.name}_star" conditioned on the new data.

    """

    def _build_conditional(Xnew, f, cov, jitter):
        if not isinstance(cov.owner.op, GPCovariance):
            raise NotImplementedError(f"Cannot build conditional of {cov.owner.op} operation")
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

    if isinstance(gp, Variable):
        assert model[gp.name] is gp
    else:
        gp = model[gp.name]

    fgraph, memo = fgraph_from_model(model)
    gp_model_var = memo[gp]
    gp_rv = gp_model_var.owner.inputs[0]

    if isinstance(gp_rv.owner.op, pm.MvNormal.rv_type):
        _, cov = gp_rv.owner.op.dist_params(gp.owner)
    else:
        raise NotImplementedError("Can only condition on pure GPs")

    # TODO: We should write the naive conditional covariance, and then have rewrites that lift it through kernels
    mu_star, cov_star = _build_conditional(Xnew, gp_model_var, cov, jitter)
    gp_rv_star = pm.MvNormal.dist(mu_star, cov_star, name=f"{gp.name}_star")

    value = gp_rv_star.clone()
    transform = None
    gp_model_var_star = model_free_rv(gp_rv_star, value, transform, *dims)

    if inline:
        fgraph.replace(gp_model_var, gp_model_var_star, import_missing=True)
    else:
        fgraph.add_output(gp_model_var_star, import_missing=True)

    return model_from_fgraph(fgraph, mutate_fgraph=True)


# @register_canonicalize
# @node_rewriter(tracks=[pm.MvNormal.rv_type])
# def GP_normal_mvnormal_conjugacy(fgraph: FunctionGraph, node):
#     # TODO: Should this alert users that it can't be applied when the GP is in a deterministic?
#     gp_rng, gp_size, mu, cov = node.inputs
#     next_gp_rng, gp_rv = node.outputs
#
#     if not isinstance(cov.owner.op, GPCovariance):
#         return
#
#     for client, input_index in fgraph.clients[gp_rv]:
#         # input_index is 2 because it goes (rng, size, mu, sigma), and we want the mu
#         # to be the GP we're looking
#         if isinstance(client.op, pm.Normal.rv_type) and (input_index == 2):
#             next_normal_rng, normal_rv = client.outputs
#             normal_rng, normal_size, mu, sigma = client.inputs
#
#             if normal_rv.ndim != gp_rv.ndim:
#                 return
#
#             X = cov.owner.inputs[0]
#
#             white_noise = WhiteNoiseCov.build_covariance(X, sigma)
#             white_noise.name = 'WhiteNoiseCov'
#             cov = cov + white_noise
#
#             if not rv_size_is_none(normal_size):
#                 normal_size = tuple(normal_size)
#                 new_gp_size = normal_size[:-1]
#                 core_shape = normal_size[-1]
#
#                 cov_shape = (*(None,) * (cov.ndim - 2), core_shape, core_shape)
#                 cov = pt.specify_shape(cov, cov_shape)
#
#             else:
#                 new_gp_size = None
#
#             next_new_gp_rng, new_gp_mvn = pm.MvNormal.dist(cov=cov, rng=gp_rng, size=new_gp_size).owner.outputs
#             new_gp_mvn.name = 'NewGPMvn'
#
#             # Check that the new shape is at least as specific as the shape we are replacing
#             for new_shape, old_shape in zip(new_gp_mvn.type.shape, normal_rv.type.shape, strict=True):
#                 if new_shape is None:
#                     assert old_shape is None
#
#             return {
#                 next_normal_rng: next_new_gp_rng,
#                 normal_rv: new_gp_mvn,
#                 next_gp_rng: next_new_gp_rng
#             }
#
#         else:
#             return None
#
# #TODO: Why do I need to register this twice?
# specialization_ir_rewrites_db.register(
#     GP_normal_mvnormal_conjugacy.__name__,
#     GP_normal_mvnormal_conjugacy,
#     "basic",
# )

# @node_rewriter(tracks=[pm.MvNormal.rv_type])
# def GP_normal_marginal_logp(fgraph: FunctionGraph, node):
#     """
#     Replace Normal(GP(cov), sigma) -> MvNormal(0, cov + diag(sigma)).
#     """
#     rng, size, mu, cov = node.inputs
#     if cov.owner and cov.owner.op == matrix_inverse:
#         tau = cov.owner.inputs[0]
#         return PrecisionMvNormalRV.rv_op(mu, tau, size=size, rng=rng).owner.outputs
#     return None
#

# cov_op = GPCovariance()
# gp_op = GP("vanilla")
# # SymbolicRandomVariable.register(type(gp_op))
# prior_from_gp = PriorFromGP()
#
# MeasurableVariable.register(type(prior_from_gp))
#
#
# @_get_measurable_outputs.register(type(prior_from_gp))
# def gp_measurable_outputs(op, node):
#     return node.outputs
