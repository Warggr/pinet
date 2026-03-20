"""Cross constraint module."""

from dataclasses import dataclass

import jax.numpy as jnp
from jax import lax

from pinet.dataclasses import ProjectionInstance

from .base import Constraint


@dataclass
class CrossConstraintParams:
    """Cross constraint set.

    The cross constraint set is defined as:
    z := x[idxs_z]
    w := M z - q
    <z, w> = 0
    z >= 0
    w >= 0
    """

    M: jnp.ndarray
    q: jnp.ndarray
    idxs_z: slice

    def __post_init__(self):
        """Check dimensions of parameters."""
        assert (
            self.M.ndim == 3
        ), "M is a matrix with shape (batch_size, n_constraints, dimension)."


class CrossConstraint(Constraint):
    """Lifted cross constraint set.

    Defined by:
    z := x[idxs_z]
    w := x[idxs_w]
    <z, w> = 0
    z >= 0
    w >= 0
    It is assumed that w are additional variables from lifting,
    and that the constraint w = M z - q is enforced somewhere else.
    """

    def __init__(
        self,
        idxs_z: slice,
        idxs_w: slice,
    ) -> None:
        """Initialize the equality constraint.

        Args:
            idxs_z: the indices of y to be considered, i.e. z = y[idxs_z]
            idxs_w: idem.
        """
        self.idxs_z = idxs_z
        self.idxs_w = idxs_w

        def simple_slice_len(sl: slice):
            assert sl.step in (1, None)
            assert sl.stop >= 0 and sl.start >= 0
            return sl.stop - sl.start

        assert simple_slice_len(self.idxs_w) == simple_slice_len(self.idxs_z)

    def project(self, yraw: ProjectionInstance) -> ProjectionInstance:
        """Project onto constraints.

        Args:
            yraw (ProjectionInstance): ProjectionInstance to projection.
                The .x attribute is the point to project.

        Returns:
            ProjectionInstance: The projected point for each point in the batch.
        """
        w = yraw.x[:, self.idxs_w]
        z = yraw.x[:, self.idxs_z]
        zero = jnp.zeros_like(w)
        _max = lax.max(lax.max(w, z), zero)
        w = jnp.select(_max == 0, zero, w)
        z = jnp.select(_max == 0, zero, z)
        w = jnp.select(_max == z, zero, w)
        z = jnp.select(_max == w, zero, z)

        return yraw.update(x=yraw.x.at[:, self.idxs_w].set(w).at[:, self.idxs_z].set(z))

    @property
    def dim(self) -> int:
        """Return the dimension of the constraint set."""
        return self.idxs_w.stop - self.idxs_z.start

    @property
    def n_constraints(self) -> int:
        """Return the number of constraints.

        If w has size n, then there are 3*n constraints:
        - z_i >= 0      forall n
        - w_i >= 0      forall n
        - z_i w_i = 0   forall n
        """
        return 3 * (self.idxs_w.stop - self.idxs_z.start)

    def cv(self, yraw: ProjectionInstance) -> jnp.ndarray:
        """Compute the constraint violation.

        Args:
            inp (ProjectionInstance): ProjectionInstance to evaluate.

        Returns:
            jnp.ndarray: The constraint violation for each point in the batch.
                Shape (batch_size, 1, 1).
        """
        w = yraw.x[:, self.idxs_w]
        z = yraw.x[:, self.idxs_z]

        return jnp.concatenate(
            [
                lax.max(-w, 0),
                lax.max(-z, 0),
                jnp.abs(w * z),
            ],
            axis=1,
        )
