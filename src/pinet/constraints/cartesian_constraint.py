"""Cartesian constraint module for combining Box and SOC constraints."""

from typing import Optional, Sequence

import jax.numpy as jnp

from pinet.constraints.cross import CrossConstraint
from pinet.dataclasses import ProjectionInstance

from .base import Constraint
from .box import BoxConstraint


class CartesianConstraint(Constraint):
    """Cartesian product of Box and Cross constraints.

    This class combines multiple Box and cross constraints that act on disjoint
    subsets of the variables (non-overlapping masks).
    """

    def __init__(
        self,
        box_constraint: Optional[BoxConstraint] = None,
        cross_constraints: Sequence[CrossConstraint] = (),
    ) -> None:
        """Initialize the Cartesian constraint.

        Args:
            box_constraint (Optional[BoxConstraint]):
                Box constraint for the Cartesian product. Defaults to None.
            nl_constraints (Optional[List[NonLinearConstraint]]):
                List of non-linear constraints (SocConstraint). Defaults to None.

        Raises:
            ValueError: If no constraints are provided, if masks overlap,
                or if constraint dimensions are inconsistent.
            TypeError: If nl_constraints is not a list or tuple.
        """
        self.box_constraint = box_constraint
        self.nl_constraints = cross_constraints

        # Validate nl_constraints
        if not isinstance(cross_constraints, (list, tuple)):
            raise TypeError(
                f"nl_constraints must be a list or tuple, "
                f"got {type(cross_constraints).__name__}."
            )

        self.constraints = [
            c for c in (box_constraint, *cross_constraints) if c is not None
        ]

        if not self.constraints:
            raise ValueError("At least one constraint must be provided.")

        # Get dimension from the first constraint
        self._dim = self.constraints[0].dim

        # Check that the constraints are boxes and socs
        if self.box_constraint is not None and not isinstance(
            self.box_constraint, BoxConstraint
        ):
            raise ValueError(
                f"The box_constraint must be a BoxConstraint, "
                f"got {type(self.box_constraint).__name__}."
            )
        if self.nl_constraints is not None:
            for constraint in self.nl_constraints:
                if not isinstance(constraint, CrossConstraint):
                    raise ValueError(
                        f"Only CrossConstraint are currently supported "
                        f"in nl_constraints, got {type(constraint).__name__}."
                    )

        # Validate that all constraints have the same dimension
        for constraint in self.constraints:
            if constraint.dim != self._dim:
                raise ValueError(
                    f"All constraints must have the same dimension. "
                    f"Expected {self._dim}, got {constraint.dim}."
                )

        # Check that masks don't overlap
        self._mask = self._validate_masks()

        self.n_nonlinear = len(self.nl_constraints)

    @property
    def mask(self):
        """Identifies which elements of the vector are affected by these Constraints."""
        return self._mask

    def _validate_masks(self) -> None:
        """Validate that masks from all constraints do not overlap.

        Raises:
            ValueError: If any masks overlap.

        Returns:
            used_mask: the union of all constraint masks
        """
        # Create a mask to track which dimensions are already used
        used_mask = jnp.zeros(self._dim, dtype=bool)

        for constraint in self.constraints:
            if isinstance(constraint, BoxConstraint):
                if jnp.any(jnp.logical_and(used_mask, constraint.mask)):
                    raise ValueError(
                        "Constraint masks overlap with previously defined constraints."
                    )
                used_mask = jnp.logical_or(constraint.mask, used_mask)
            else:
                mask = constraint.get_mask()
                if jnp.any(jnp.logical_and(used_mask, mask)):
                    raise ValueError(
                        "Constraint masks overlap with previously defined constraints."
                    )
                used_mask = jnp.logical_or(used_mask, mask)
        return used_mask

    def project(self, yraw: ProjectionInstance) -> ProjectionInstance:
        """Project the input to the feasible region.

        Projects onto each constraint independently. Since masks don't overlap,
        each constraint operates on a disjoint subset of variables.

        Args:
            yraw (ProjectionInstance): ProjectionInstance to project.

        Returns:
            ProjectionInstance: The projected input.
        """
        if self.box_constraint is not None:
            yraw = self.box_constraint.project(yraw)

        # Project onto each constraint
        for nl_constraint in self.nl_constraints:
            yraw = nl_constraint.project(yraw)

        return yraw

    def cv(self, yraw: ProjectionInstance) -> jnp.ndarray:
        """Compute the constraint violation.

        Returns the maximum constraint violation across all constraints.

        Args:
            yraw (ProjectionInstance): ProjectionInstance to evaluate.

        Returns:
            jnp.ndarray: The constraint violation for each point in the batch.
                Shape (batch_size, 1, 1).
        """
        cvs = []

        if self.box_constraint is not None:
            cvs.append(self.box_constraint.cv(yraw).reshape(-1))

        # Compute constraint violations for nl constraints
        for constraint in self.nl_constraints:
            cvs.append(constraint.cv(yraw).reshape(-1))

        # Return the maximum violation
        if len(cvs) == 1:
            return cvs[0]
        else:
            return jnp.max(jnp.array(cvs), 0).reshape(-1, 1, 1)

    @property
    def dim(self) -> int:
        """Return the dimension of the constraint set.

        Returns:
            int: The dimension of the constraint set.
        """
        return self._dim

    @property
    def n_constraints(self) -> int:
        """Return the total number of constraints.

        Sums up the number of constraints from all constituent constraints.

        Returns:
            int: The total number of constraints.
        """
        total = 0
        for constraint in self.constraints:
            total += constraint.n_constraints
        return total
