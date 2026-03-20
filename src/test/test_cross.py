"""Tests for the cross constraint."""

import jax.numpy as jnp

from pinet import CrossConstraint, ProjectionInstance


def test_cross_constraint_cv():
    x = jnp.expand_dims(jnp.array([1, 1, -2, 2]), 0)
    c = CrossConstraint(idxs_z=slice(0, 2), idxs_w=slice(2, 4))
    inst = ProjectionInstance(x=x)

    assert jnp.all(
        c.cv(inst)
        == jnp.expand_dims(
            jnp.array(
                [
                    2,  # w >= 0
                    0,
                    0,  # z >= 0
                    0,
                    2,  # w * z = 0
                    2,
                ]
            ),
            axis=0,
        )
    )
    assert jnp.all(
        c.project(inst).x
        == jnp.expand_dims(
            jnp.array(
                [
                    1,
                    0,
                    0,
                    2,
                ]
            ),
            axis=0,
        )
    )


if __name__ == "__main__":
    test_cross_constraint_cv()
