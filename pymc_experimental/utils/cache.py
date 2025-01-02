import hashlib
import os
import sys
from typing import Callable, Literal

import arviz as az
import numpy as np
import pymc
import pytensor
from pymc import (
    modelcontext,
    sample,
    sample_posterior_predictive,
    sample_prior_predictive,
)
from pymc.model.fgraph import fgraph_from_model
from pytensor.compile import SharedVariable
from pytensor.graph import Constant, FunctionGraph, Variable
from pytensor.scalar import ScalarType
from pytensor.tensor import TensorType
from pytensor.tensor.random.type import RandomType
from pytensor.tensor.type_other import NoneTypeT

import pymc_experimental


def hash_data(c: Variable) -> str:
    if isinstance(c.type, NoneTypeT):
        return "None"
    if isinstance(c.type, (ScalarType, TensorType)):
        if isinstance(c, Constant):
            arr = c.data
        elif isinstance(c, SharedVariable):
            arr = c.get_value(borrow=True)
        arr_data = arr.view(np.uint8) if arr.size > 1 else arr.tobytes()
        return hashlib.sha1(arr_data).hexdigest()
    else:
        raise NotImplementedError(f"Hashing not implemented for type {c.type}")


def get_name_and_props(obj):
    name = str(obj)
    props = str(getattr(obj, "_props", lambda: {})())
    return name, props


def hash_from_fg(fg: FunctionGraph) -> str:
    objects_to_hash = []
    for node in fg.toposort():
        objects_to_hash.append(
            (
                get_name_and_props(node.op),
                tuple(get_name_and_props(inp.type) for inp in node.inputs),
                tuple(get_name_and_props(out.type) for out in node.outputs),
                # Name is not a symbolic input in the fgraph representation, maybe it should?
                tuple(inp.name for inp in node.inputs if inp.name),
                tuple(out.name for out in node.outputs if out.name),
            )
        )
        objects_to_hash.append(
            tuple(
                hash_data(c)
                for c in node.inputs
                if (
                    isinstance(c, (Constant, SharedVariable))
                    # Ignore RNG values
                    and not isinstance(c.type, RandomType)
                )
            )
        )
    str_hash = "\n".join(map(str, objects_to_hash))
    return hashlib.sha1(str_hash.encode()).hexdigest()


def cache_sampling(
    sampling_fn: Literal[sample, sample_prior_predictive, sample_posterior_predictive],
    dir: str = "",
    force_sample: bool = False,
    force_load: bool = True,
) -> Callable:
    """Cache the result of PyMC sampling.

    Parameter
    ---------
    sampling_fn: Callable
        Must be one of `pymc.sample`, `pymc.sample_prior_predictive` or `pymc.sample_posterior_predictive`.
        Positional arguments are disallowed.
    dir: string, Optional
        The directory where the results should be saved or retrieved from. Defaults to working directory.
    force_sample: bool, Optional
        Whether to force sampling even if cache is found. Defaults to False.
    force_load:

    Returns
    -------
    cached_sampling_fn: Callable
        Function that wraps the sampling_fn. When called, the wrapped function will look for a valid cached result.
        A valid cache requires the same:
           1. Model and data
           2. Sampling function
           3. Sampling kwargs, ignoring ``random_seed``, ``trace``, ``progressbar``, ``extend_inferencedata`` and ``compile_kwargs``.
           4. PyMC, PyTensor, and PyMC-Experimental versions
        If a valid cache is found, sampling is bypassed altogether, unless ``force_sample=True``.
        Otherwise, sampling is performed and the result cached for future reuse.
        Caching is done on the basis of SHA-1 hashing, and there could be unlikely false positives.


    Examples
    --------

    .. code-block:: python

        import pymc as pm
        from pymc_experimental.utils.cache import cache_sampling

        with pm.Model() as m:
            y_data = pm.MutableData("y_data", [0, 1, 2])
            x = pm.Normal("x", 0, 1)
            y = pm.Normal("y", mu=x, observed=y_data)

            cache_sample = cache_sampling(pm.sample, dir="traces")
            idata1 = cache_sample(chains=2)

            # Cache hit! Returning stored result
            idata2 = cache_sample(chains=2)

            pm.set_data({"y_data": [1, 1, 1]})
            idata3 = cache_sample(chains=2)

        assert idata1.posterior["x"].mean() == idata2.posterior["x"].mean()
        assert idata1.posterior["x"].mean() != idata3.posterior["x"].mean()

    """
    allowed_fns = (sample, sample_prior_predictive, sample_posterior_predictive)
    if sampling_fn not in allowed_fns:
        raise ValueError(f"Cache sampling can only be used with {allowed_fns}")

    def wrapped_sampling_fn(*args, model=None, random_seed=None, **kwargs):
        if args:
            raise ValueError("Non-keyword arguments not allowed in cache_sampling")

        extend_inferencedata = kwargs.pop("extend_inferencedata", False)

        # Model hash
        model = modelcontext(model)
        fg, _ = fgraph_from_model(model)
        model_hash = hash_from_fg(fg)

        # Sampling hash
        sampling_hash_dict = kwargs.copy()
        sampling_hash_dict.pop("trace", None)
        sampling_hash_dict.pop("random_seed", None)
        sampling_hash_dict.pop("progressbar", None)
        sampling_hash_dict.pop("compile_kwargs", None)
        sampling_hash_dict["sampling_fn"] = str(sampling_fn)
        sampling_hash_dict["versions"] = (
            pymc.__version__,
            pytensor.__version__,
            pymc_experimental.__version__,
        )
        sampling_hash = str(sampling_hash_dict)

        file_name = hashlib.sha1((model_hash + sampling_hash).encode()).hexdigest() + ".nc"
        file_path = os.path.join(dir, file_name)

        if not force_sample and os.path.exists(file_path):
            print("Cache hit! Returning stored result", file=sys.stdout)
            idata_out: az.InferenceData = az.from_netcdf(file_path)
            if force_load:
                idata_out.load()

        else:
            idata_out = sampling_fn(*args, **kwargs, model=model, random_seed=random_seed)
            if os.path.exists(file_path):
                os.remove(file_path)
            if not os.path.exists(dir):
                os.mkdir(dir)
            az.to_netcdf(idata_out, file_path)

        # We save inferencedata separately and extend if needed
        if extend_inferencedata:
            trace = kwargs["trace"]
            trace.extend(idata_out)
            idata_out = trace

        return idata_out

    return wrapped_sampling_fn
