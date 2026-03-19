"""Cross constraint module."""

import jax.numpy as jnp
from jax import lax

from pinet.dataclasses import ProjectionInstance

from .base import Constraint


class CrossConstraint(Constraint):
    """Cross constraint set.

    The cross constraint set is defined as:
    z := x[idxs_z]
    w := M z - q
    <z, w> = 0
    z >= 0
    w >= 0
    """

    def __init__(
        self,
        M: jnp.ndarray,
        q: jnp.ndarray,
        idxs_z: slice,
    ) -> None:
        """Initialize the equality constraint.

        Args:
            M (jnp.ndarray): Left hand side matrix.
                Shape (batch_size, n_constraints, dimension).
            q (jnp.ndarray): Right hand side vector.
                Shape (batch_size, n_constraints, 1).
            idxs_z: the indices of y to be considered, i.e. z = y[idxs_z]
        """
        assert M is not None, "Matrix A must be provided."

        self.M = M
        self.q = q
        self.idxs_z = idxs_z
        self.setup()

    def setup(self) -> None:
        """Sets up the equality constraint."""
        assert (
            self.M.ndim == 3
        ), "M is a matrix with shape (batch_size, n_constraints, dimension)."

    def project(self, yraw: ProjectionInstance) -> ProjectionInstance:
        """Project onto constraints.

        Args:
            yraw (ProjectionInstance): ProjectionInstance to projection.
                The .x attribute is the point to project.

        Returns:
            ProjectionInstance: The projected point for each point in the batch.
        """

        def project_unbatched(x):
            # The unbatched version is necessary
            # because lax.switch has no batched version.
            from jax.numpy import dot

            def case_1(w, z):
                from jax.numpy.linalg import norm

                lambd = (dot(w, w) + dot(z, z) - norm(w + z) * norm(w - z)) / (
                    2 * dot(w, z)
                )
                factor = 1 / (1 - lambd**2)
                return factor * (w - lambd * z), factor * (z - lambd * w)

            def case_2(w, z):
                return jnp.zeros_like(w), z

            z = x[self.idxs_z]
            w = self.M @ z - self.q
            is_zero = dot(w, z) == 0
            norm_is_equal = jnp.all(w == z) | jnp.all(w == -z)
            cond = jnp.where(is_zero, 0, jnp.where(norm_is_equal, 2, 1))
            w, z = lax.switch(
                cond,
                [
                    lambda w, z: (w, z),
                    case_1,
                    case_2,
                ],
                w,
                z,
            )
            return w, z

        w, z = project_unbatched(yraw.x[0])

        return yraw.update(x=yraw.x.at[:, self.idxs_w].set(w).at[:, self.idxs_z].set(z))

    @property
    def dim(self) -> int:
        """Return the dimension of the constraint set."""
        return self.A.shape[-1]

    @property
    def n_constraints(self) -> int:
        """Return the number of constraints."""
        return self.A.shape[1]

    def cv(self, inp: ProjectionInstance) -> jnp.ndarray:
        """Compute the constraint violation.

        Args:
            inp (ProjectionInstance): ProjectionInstance to evaluate.

        Returns:
            jnp.ndarray: The constraint violation for each point in the batch.
                Shape (batch_size, 1, 1).
        """
        b, A, _ = self.get_params(inp)

        return jnp.linalg.norm(A @ inp.x - b, ord=jnp.inf, axis=1, keepdims=True)
