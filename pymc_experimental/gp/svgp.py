import pytensor
import pytensor.tensor as pt
import pymc as pm
import numpy as np

import optax
import jax
import jax.random as jr

from pymc.gp.util import stabilize
from pymc.sampling.jax import get_jaxified_graph
from functools import partial


class SVGP:
    def __init__(
        self,
        input_dim,
        n_data,
        batch_size,
        mean_func,
        cov_func,
        sigma,
        z_init,
        variational_sd_dist,
        jitter=1e-6,
    ):
        self.mean_func = mean_func
        self.cov_func = cov_func
        self.sigma = sigma
        self.jitter = jitter

        self.input_dim = input_dim
        self.n_inducing = z_init.shape[0]
        self.n_data = n_data
        self.batch_size = batch_size

        self.z_init = z_init
        self.variational_sd_dist = variational_sd_dist
        
        self.initialize()

    def initialize(self, model=None):
        with pm.modelcontext(model):
            
            self.z = pm.Flat("z", shape=(self.n_inducing, self.input_dim), initval=self.z_init)
            self.variational_mean = pm.Flat("variational_mean", shape=self.n_inducing)
            variational_root_chol, _, _ = pm.LKJCholeskyCov(
                "vrc", n=self.n_inducing, eta=1.0, sd_dist=self.variational_sd_dist,
            )
            self.variational_root_covariance = variational_root_chol @ variational_root_chol.T

    def kl_divergence(self):
        mu = self.variational_mean
        sqrt = self.variational_root_covariance
        z = self.z

        muz = self.mean_func(z)
        Kzz = self.cov_func(z)
        return self.kl_mvn(mu, sqrt @ sqrt.T, muz, Kzz)
    
    @staticmethod
    def kl_mvn(mu1, K1, mu2, K2):
        # TODO, rewrite to tale in cholesky L1 instead of K1
        d = mu2 - mu1

        K1 = stabilize(K1)
        K2 = stabilize(K2)

        L1 = pt.linalg.cholesky(K1, lower=True)
        L2 = pt.linalg.cholesky(K2, lower=True)

        logdet1 = 2 * pt.sum(pt.log(pt.diag(L1)))
        logdet2 = 2 * pt.sum(pt.log(pt.diag(L2)))

        def solve(B):
            return pt.linalg.solve_triangular(
                L2.T, pt.linalg.solve_triangular(L2, B, lower=True), lower=False
            )

        term1 = pt.trace(solve(K1))
        term2 = logdet2 - logdet1
        term3 = d.T @ solve(d)
        return (term1 + term2 + term3 - d.shape[0]) / 2.0

    def predict(self, t, sigma=None):
        mu = self.variational_mean
        sqrt = self.variational_root_covariance

        muz = self.mean_func(self.z)
        Kzz = stabilize(self.cov_func(self.z))
        Lz = pt.linalg.cholesky(Kzz)

        Ktt = stabilize(self.cov_func(t))
        
        mut = self.mean_func(t)
        Kzt = self.cov_func(self.z, t)
        
        Lz_inv_Kzt = pt.linalg.solve_triangular(Lz, Kzt, lower=True)  # Lz⁻¹ Kzt
        Kzz_inv_Kzt = pt.linalg.solve_triangular(Lz.mT, Lz_inv_Kzt, lower=False)  # Kzz⁻¹ Kzt
        Ktz_Kzz_inv_sqrt = pt.matmul(Kzz_inv_Kzt.mT, sqrt)  # Ktz Kzz⁻¹ sqrt

        mean = mut + pt.matmul(Kzz_inv_Kzt.mT, mu - muz)  # μt + Ktz Kzz⁻¹ (μ - μz)

        if sigma is None:
            noise = (1e-6)**2 * pt.identity_like(Ktt)
        else:
            noise = sigma**2 * pt.identity_like(Ktt)
        
        covariance = (
            Ktt
            - pt.matmul(Lz_inv_Kzt.mT, Lz_inv_Kzt)
            + pt.matmul(Ktz_Kzz_inv_sqrt, Ktz_Kzz_inv_sqrt.mT)
            + noise
        )
        return mean, covariance
    
    def variational_expectation(self, X_batch, y_batch):

        X_batch = pt.as_tensor(X_batch)
        
        def diag_predict(X_batch):
            mean, cov = self.predict(X_batch)
            return mean, pt.diag(cov)

        func = pt.vectorize(diag_predict, "(o, k) -> (o), (o)")
        mean, variance = func(pt.expand_dims(X_batch, -2))
        
        ## integrate expectation
        sq_error = pt.square(y_batch - mean)
        expectation = -0.5 * pt.sum(
            pt.log(2.0 * pt.pi) + pt.log(self.sigma**2) + (sq_error + variance) / self.sigma**2, axis=1
        )
        return expectation

    def elbo(self, X_batch, y_batch):
        var_exp = self.variational_expectation(X_batch, y_batch)
        #n, b = X.shape[0].eval(), X.shape[0]
        return (self.n_data / self.batch_size) * pt.sum(var_exp).squeeze() - self.kl_divergence()

    def fit(self, X_data, y_data, optimizer, params=None, n_steps=100_000, model=None):
        
        if X_data.ndim != 2 or y_data.ndim != 2:
            raise ValueError("no")
        
        with pm.modelcontext(model) as model:
            loss = -self.elbo(model["X"], model["y"])
            training_step = make_training_step_fn(model, loss, optimizer, self.input_dim, batch_size=self.batch_size)

        if params is None:
            initial_point = model.initial_point()
            params = tuple(initial_point.values())
        
        optimizer_state = optimizer.init(params)
        var_names = model.initial_point().keys()

        loss_history = []
        for step in range(n_steps):
            try:
                batch_slice = np.random.choice(self.n_data, size=self.batch_size, replace=False)
                params, optimizer_state, loss_value = training_step(
                    X_data[batch_slice, :],
                    y_data[batch_slice, :],
                    params,
                    optimizer_state,
                )
                if (len(loss_history) > 1) and (loss_value < loss_history[-1]):
                    best_params = params
                
                loss_history.append(loss_value)
    
                if step % 100 == 0:
                    print(f"Iteration: {step}, Loss: {loss_value:.2f}", end="\r")
            
            except KeyboardInterrupt:
                break
            
        print(f"Iteration: {step + 1}, Loss: {loss_value:.2f}, finished.", end="\r")
        return best_params, loss_history

    @staticmethod
    def get_batch(X, y, n, batch_size: int, key):
        # Subsample mini-batch indices with replacement.
        indices = jr.choice(key, n, (batch_size,), replace=True)
        return X[indices, :], y[indices]
    
    def fit_scan(
        self, 
        X_data, 
        y_data,
        optimizer, 
        params=None,
        num_iters=100_000, 
        model=None,
        unroll=1,
        key = jr.PRNGKey(42)
    ):

        X_data = jax.numpy.array(X_data)
        y_data = jax.numpy.array(y_data)
        
        with pm.modelcontext(model) as model:
            loss = -self.elbo(model["X"], model["y"])

        point = model.initial_point()
        [loss_w_values] = model.replace_rvs_by_values([loss])
        # [loss2], joined_inputs = pm.pytensorf.join_nonshared_inputs(
        #    point=point, outputs=[loss_w_values], inputs=model.continuous_value_vars# + pm.inputvars(loss) # for pt.tensor
        # )
        # replace X, y with their minibatch
        X_batch = pt.tensor("X_batch", shape=(self.batch_size, self.input_dim))
        y_batch = pt.tensor("y_batch", shape=(self.batch_size, 1))
    
        X, y = model["X"], model["y"]
        loss2 = pytensor.graph.graph_replace(
            loss_w_values,
            replace={
                X: X_batch,
                y: y_batch,
            },
        )
    
        f_loss_jax = get_jaxified_graph(
            [X_batch, y_batch, *model.continuous_value_vars], outputs=[loss2]
        )

        def f_loss(X, y, params):
            return f_loss_jax(X, y, *params)[0]
        
        if params is None:
            initial_point = model.initial_point()
            params = tuple(initial_point.values())
        
        optimizer_state = optimizer.init(params)
        var_names = model.initial_point().keys()

        # Initialise optimiser state.
        opt_state = optimizer.init(params)
    
        # Mini-batch random keys to scan over.
        iter_keys = jr.split(key, num_iters)
        
        def step(carry, key):
            params, opt_state = carry
            X_data_batch, y_data_batch = self.get_batch(X_data, y_data, self.n_data, self.batch_size, key)
            loss_val, loss_grads = jax.value_and_grad(f_loss, 2)(X_data_batch, y_data_batch, params)
            updates, opt_state = optimizer.update(loss_grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss_val

        (params, _), history = jax.lax.scan(step, (params, opt_state), (iter_keys), unroll=unroll)
        return params, history
        
    
    def compile_pred_func(self, sigma=None, diag=False, mode="FAST_RUN", model=None):
        t = pt.tensor("t", shape=(None, self.input_dim))

        if diag:
            def diag_predict(X):
                mean, cov = self.predict(X, sigma=sigma)
                return mean, pt.diag(cov)
            
            func = pt.vectorize(diag_predict, "(o, k) -> (o), (o)")
            #mu, cov = func(t[..., None]) # cov is actually a variance
            mu, cov = func(pt.expand_dims(t, -2))
        else:
            mu, cov = self.predict(t, sigma=sigma)
       
        
        with pm.modelcontext(model) as model:
            mu_value, cov_value = model.replace_rvs_by_values([mu, cov])
        
        inputs = pm.inputvars([mu_value, cov_value])
        f_predict = pytensor.function(
            inputs=inputs,
            outputs=[mu_value.squeeze(), cov_value.squeeze()],
            on_unused_input="ignore",
            mode=mode,
        )
        return partial(
            self._predict_f,
            inputs=inputs,
            f_predict=f_predict,
        )

    def _predict_f(self, X_pred, result_dict, inputs, f_predict):
        input_names = [x.name for x in inputs]
        mu_pred, cov_pred = f_predict(
            **{k: v for k, v in result_dict.items() if k in input_names}, t=X_pred
        )
        return mu_pred, cov_pred


def make_training_step_fn(
    model,
    loss,
    optimizer,
    input_dim,
    batch_size=512,
    n_devices=1,
):
    point = model.initial_point()
    [loss_w_values] = model.replace_rvs_by_values([loss])
    # [loss2], joined_inputs = pm.pytensorf.join_nonshared_inputs(
    #    point=point, outputs=[loss_w_values], inputs=model.continuous_value_vars# + pm.inputvars(loss) # for pt.tensor
    # )
    # replace X, y with their minibatch
    X_batch = pt.tensor("X_batch", shape=(batch_size, input_dim))
    y_batch = pt.tensor("y_batch", shape=(batch_size, 1))

    X, y = model["X"], model["y"]
    loss2 = pytensor.graph.graph_replace(
        loss_w_values,
        replace={
            X: X_batch,
            y: y_batch,
        },
    )

    
    # to have in pytensor not jax:
    # loss2 = pymc.pytensorf.rewrite_pregrad(loss2) 
    # grad = pt.grad(loss2, model.continue_value_vars)
    #  f_value_and_grad = pytensor.function(inputs=[X_batch, ...], ouputs = [loss2, grad], **compile_kwargs)
    
    f_loss_jax = get_jaxified_graph(
        [X_batch, y_batch, *model.continuous_value_vars], outputs=[loss2]
    )
    #f_loss_jax = pytensor.function(
    #    [X_batch, y_batch, *model.continuous_value_vars],
    #    outputs=[loss2],
    #    mode='JAX'
    #)

    def f_loss(X, y, params):
        #print("two", X.shape, y.shape, len(params))
        return f_loss_jax(X, y, *params)[0]

    # @partial(jax.pmap, axis_name="device")
    @jax.jit
    def training_step(X, y, params, optimizer_state):
        
        #print("one", X.shape, y.shape)
        loss, grads = jax.value_and_grad(f_loss, 2)(X, y, params)

        ## with partial(jax.pmap), comment if that decor is gone
        # loss = jax.lax.psum(loss, axis_name="device")
        # grads = jax.lax.psum(grads, axis_name="device")

        updates, optimizer_state = optimizer.update(grads, optimizer_state, params)

        params = optax.apply_updates(params, updates)
        return params, optimizer_state, loss

    return training_step