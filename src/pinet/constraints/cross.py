"""Cross constraint module."""

from typing import Literal

import jax.numpy as jnp
from jax import lax
from vector_structure import simple_slice_len

from pinet.dataclasses import ProjectionInstance

from .base import Constraint


class CrossConstraint(Constraint):
    """Cross constraint set.

    The cross constraint set is defined as:
    z := x[idxs_z]
    w := M x[idxs_w] - q
    <z, w> = 0
    z >= 0
    w >= 0

    """

    def __init__(
        self,
        idxs_z: slice,
        idxs_w: slice,
        M: jnp.ndarray | Literal[1] = 1,
        q: jnp.ndarray | Literal[0] = 0,
        dim: int | None = None,
    ):
        """Constructor.

        Attributes:
            M: shape (batch_size, n_z, n_w)
            q: shape (batch_size, n_z, 1) (for consistency with AffineEquality)
            idxs_w: slice of size n_w
            idxs_z: slice of size n_z
        """
        self.idxs_w = idxs_w
        self.idxs_z = idxs_z
        self._dim = dim
        if isinstance(M, (float, int)) and M == 1:
            self.M = None
        else:
            self.M = M
        if isinstance(q, (float, int)) and q == 0:
            self.q = None
        else:
            self.q = q

        if self.M is not None:
            assert (
                self.M.ndim == 3
            ), "M is a matrix with shape (batch_size, n_constraints, dimension)."
            assert self.M.shape[1] == simple_slice_len(self.idxs_z)
            assert self.M.shape[2] == simple_slice_len(self.idxs_w)
        else:
            assert simple_slice_len(self.idxs_w) == simple_slice_len(self.idxs_z)
        if self.q is not None:
            assert (
                self.q.ndim == 3
            ), "q is an array with shape (batch_size, n_constraints, 1)."
            assert self.q.shape[1] == simple_slice_len(self.idxs_z)
            assert self.q.shape[2] == 1

    @property
    def dim(self) -> int:
        """Returns the dimension of the constraint set."""
        if self._dim is None:
            raise ValueError(
                "_dim must be provided to CrossConstraint before it can be used"
            )
        return self._dim

    def num_auxiliary_variables(self):
        """The number of auxiliary vars introduced by this constraint."""
        return simple_slice_len(self.idxs_z)

    def get_auxiliary_variables(self, y):
        """Construct the auxiliary variables.

        Arguments:
            y: the lifted vector. Shape (batch_size, dim).
        """
        w = self.M @ y[:, self.idxs_w] + self.q
        return w

    def num_constraints(self):
        """The number of constraints."""
        return simple_slice_len(self.idxs_z)

    def get_mask(self) -> jnp.ndarray:
        """Identifies which variables are affected by these constraints."""
        mask = jnp.zeros(self.dim).at[self.idxs_w].set(1).at[self.idxs_z].set(1)
        return mask

    def project(self, yraw: ProjectionInstance) -> ProjectionInstance:
        """Project onto constraints.

        Args:
            yraw (ProjectionInstance): ProjectionInstance to projection.
                The .x attribute is the point to project.

        Returns:
            ProjectionInstance: The projected point for each point in the batch.
        """
        assert self.M is None and self.q is None, "Please lift constraints first"
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

        cvs = jnp.concatenate(
            [
                lax.max(-w, 0),
                lax.max(-z, 0),
                jnp.abs(w * z),
            ],
            axis=1,
        )
        return jnp.max(cvs, axis=1).reshape((-1, 1, 1))
