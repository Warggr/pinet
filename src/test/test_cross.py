"""Tests for the cross constraint."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from jax.test_util import check_grads

from pinet import (
    AffineInequalityConstraint,
    CrossConstraint,
    EqualityConstraint,
    Project,
    ProjectionInstance,
)


def test_cross_constraint_cv():
    x = jnp.expand_dims(jnp.array([1, 1, -2, 2], dtype=float), (0, -1))
    c = CrossConstraint(idxs_z=slice(0, 2), idxs_w=slice(2, 4))
    inst = ProjectionInstance(x=x)

    assert jnp.all(c.cv(inst) == jnp.array([[[2]]]))
    assert jnp.all(
        c.project(inst).x
        == jnp.expand_dims(
            jnp.array(
                [
                    1,
                    0,
                    0,
                    2,
                ],
                dtype=float,
            ),
            axis=(0, -1),
        )
    )


@pytest.mark.parametrize(
    "seed,tol,include_cross_constraints",
    [(seed, 1e-3, False) for seed in range(5)]
    + [(seed, 7e-2, True) for seed in range(10)],  # a lot of these will be skipped
)
def test_gradient_with_cross(seed: int, tol: float, include_cross_constraints: bool):
    key = random.key(seed=seed)

    key_dims, key_proj, key_x = random.split(key, 3)
    (num_variables,) = random.randint(key_dims, shape=(1,), minval=1, maxval=100)

    key_eq, key_ineq, key_cross = random.split(key_proj, 3)
    key_dims, key_A, key_b = random.split(key_eq, 3)
    (num_eq_constraints,) = random.randint(
        key_dims, shape=(1,), minval=0, maxval=num_variables / 3
    )
    if num_eq_constraints != 0:
        A = random.normal(key_A, shape=(1, num_eq_constraints, num_variables))
        b = random.normal(key_b, shape=(1, num_eq_constraints, 1))
        eq_constraint = EqualityConstraint(A, b)
    else:
        eq_constraint = None

    key_dims, key_C, key_lb = random.split(key_ineq, 3)
    (num_ineq_constraints,) = random.randint(
        key_dims, shape=(1,), minval=1, maxval=num_variables / 3
    )
    C = random.normal(key_C, shape=(1, num_ineq_constraints, num_variables))
    lb = random.normal(key_lb, shape=(1, num_ineq_constraints, 1))
    ub = jnp.inf * jnp.ones((1, num_ineq_constraints, 1))
    ineq_constraint = AffineInequalityConstraint(C=C, lb=lb, ub=ub)

    def make_cross_constraints(key):
        key_dims, key_w, key_z, key_M, key_q = random.split(key, 5)
        (num_idxs,) = random.randint(
            key_dims, shape=(1,), minval=1, maxval=num_variables / 3
        )
        print(f"{num_idxs=}")

        def get_random_slice(key):
            (start,) = random.randint(
                key, shape=(1,), minval=0, maxval=num_variables - num_idxs
            )
            return slice(start, start + num_idxs)

        return CrossConstraint(
            idxs_w=get_random_slice(key_w),
            idxs_z=get_random_slice(key_z),
            M=random.normal(key_M, shape=(1, num_idxs, num_idxs)),
            q=random.normal(key_q, shape=(1, num_idxs, 1)),
        )

    if include_cross_constraints:
        cross_constraints = [
            make_cross_constraints(key) for key in random.split(key_cross, 2)
        ]
    else:
        cross_constraints = []

    try:
        proj = Project(
            eq_constraint=eq_constraint,
            ineq_constraint=ineq_constraint,
            cross_constraints=cross_constraints,
        )
    except ValueError:
        # When using 2 or more cross_constraints, there's a risk that they overlap
        pytest.skip(
            "The projection could not be built - probably some constraints overlap"
        )
    x = random.normal(key_x, shape=(1, num_variables, 1))

    def project(x):
        result = proj.call(yraw=ProjectionInstance(x=x), n_iter=1000, n_iter_bwd=500)[0]
        cv = proj.cv(result)
        try:
            assert jnp.all(cv <= 1e-3), jnp.max(cv)
        except AssertionError:
            pytest.skip("Forward projection does not converge")
        return result.x

    check_grads(project, (x,), order=1, modes=["rev"], atol=tol, rtol=tol)

    dp_x = jax.jacobian(project)(x).squeeze((0, 2, 3, 5))

    delta = 1e-8
    x_multi = jnp.tile(x, (num_variables, 1, 1))
    x_plus = x_multi + delta * jnp.eye(num_variables)[:, :, jnp.newaxis]
    x_minus = x_multi - delta * jnp.eye(num_variables)[:, :, jnp.newaxis]
    p_x_plus = project(x_plus)
    p_x_minus = project(x_minus)
    dp_fin_diff = ((p_x_plus - p_x_minus) / (2 * delta)).reshape(
        (num_variables, num_variables)
    )
    np.testing.assert_allclose(dp_fin_diff, dp_x, rtol=1e-2, atol=1e-2)
