"""Parser of constraints to lifted representation module."""

from typing import Callable, Optional, Sequence

import jax.numpy as jnp
import numpy as np
from vector_structure import VectorStructure, simple_slice_len

from pinet.dataclasses import BoxConstraintSpecification

from .affine_equality import EqualityConstraint
from .affine_inequality import AffineInequalityConstraint
from .box import BoxConstraint
from .cartesian_constraint import CartesianConstraint
from .cross import CrossConstraint


class ConstraintParser:
    """Parse constraints into a lifted representation.

    This class takes as input an equality, an inequality, and a box constraint.
    It returns an equivalent equality and box constraint in a lifted representation.
    """

    def __init__(
        self,
        eq_constraint: EqualityConstraint | None,
        ineq_constraint: AffineInequalityConstraint | None,
        box_constraint: BoxConstraint | None = None,
        nl_constraints: Sequence[CrossConstraint] = (),
    ) -> None:
        """Initiaze the constraint parser.

        Args:
            eq_constraint (EqualityConstraint): An equality constraint.
            ineq_constraint (AffineInequalityConstraint): An inequality constraint.
            box_constraint (BoxConstraint): A box constraint.
            nl_constraints (CrossConstraint): A cross constraint.
        """
        if ineq_constraint is None and not nl_constraints:
            # The constraints do not need lifting.
            self.parse = lambda method: (eq_constraint, box_constraint, lambda y: y)
            return

        self.n_ineq = ineq_constraint.n_constraints if ineq_constraint else 0
        if eq_constraint is None:
            eq_constraint = EqualityConstraint(
                A=jnp.empty((1, 0, self.dim)),
                b=jnp.empty((1, 0, 1)),
                method=None,
                var_b=False,
                var_A=False,
            )

        self.eq_constraint = eq_constraint
        self.n_eq = eq_constraint.n_constraints
        self.ineq_constraint = ineq_constraint
        self.box_constraint = box_constraint
        self.nl_constraints = nl_constraints
        self.n_cross = sum(lc.num_auxiliary_variables() for lc in self.nl_constraints)

        vector_structure = [("y", eq_constraint.dim)]
        vector_structure.append(("y_aux", self.n_ineq))
        for i, lc in enumerate(nl_constraints):
            vector_structure.append((f"w_{i}", lc.num_auxiliary_variables()))
        self.vector_structure = VectorStructure(vector_structure)
        self.dim = self.vector_structure.size

        # Batch consistency checks
        self.batch_size = self.eq_constraint.A.shape[0]
        if self.ineq_constraint is not None:
            assert (
                self.eq_constraint.A.shape[0] == self.ineq_constraint.C.shape[0]
                or self.eq_constraint.A.shape[0] == 1
                or self.ineq_constraint.C.shape[0] == 1
            ), "Batch sizes of A and C must be consistent."
            self.batch_size = max(self.batch_size, self.ineq_constraint.C.shape[0])
        if self.box_constraint is not None:
            assert (
                self.ineq_constraint.lb.shape[0] == self.box_constraint.lb.shape[0]
                or self.ineq_constraint.lb.shape[0] == 1
                or self.box_constraint.lb.shape[0] == 1
            ), "Batch sizes of lb and lower_bound must be consistent."

            assert (
                self.ineq_constraint.ub.shape[0] == self.box_constraint.ub.shape[0]
                or self.ineq_constraint.ub.shape[0] == 1
                or self.box_constraint.ub.shape[0] == 1
            ), "Batch sizes of ub and upper_bound must be consistent."

    def parse(
        self, method: Optional[str] = "pinv"
    ) -> tuple[EqualityConstraint, CartesianConstraint | BoxConstraint, Callable]:
        """Parse the constraints into a lifted representation.

        Args:
            method (Optional[str]): Method to use for solving linear systems.
                Valid methods are "pinv", and None.

        Returns:
            A tuple of constraints: (eq_constraint, box_constraint)
        """
        # Build lifted A matrix.
        # Maximum batch size between A and C
        mbAC = self.batch_size
        As, bs = [], []
        first_row_batched = jnp.tile(
            jnp.concatenate(
                [
                    self.eq_constraint.A,
                    jnp.zeros(
                        shape=(
                            self.eq_constraint.A.shape[0],
                            self.n_eq,
                            simple_slice_len(self.vector_structure["y_aux":]),
                        )
                    ),
                ],
                axis=2,
            ),
            (mbAC // self.eq_constraint.A.shape[0], 1, 1),
        )

        if self.ineq_constraint:
            second_row_batched = jnp.tile(
                jnp.concatenate(
                    [
                        self.ineq_constraint.C,
                        -jnp.tile(
                            jnp.eye(self.n_ineq).reshape(1, self.n_ineq, self.n_ineq),
                            (self.ineq_constraint.C.shape[0], 1, 1),
                        ),
                        jnp.zeros(
                            shape=(
                                self.ineq_constraint.C.shape[0],
                                self.n_ineq,
                                simple_slice_len(self.vector_structure["w"]),
                            )
                        ),
                    ],
                    axis=2,
                ),
                (mbAC // self.ineq_constraint.C.shape[0], 1, 1),
            )
            As.append(second_row_batched)
            bs.append(jnp.zeros(shape=(self.eq_constraint.b.shape[0], self.n_ineq, 1)))
        for i, lc in enumerate(self.nl_constraints):
            if lc.M is not None or lc.q is not None:
                batch_size = 1
                if lc.M is not None:
                    batch_size = lc.M.shape[0]
                row = jnp.zeros(
                    shape=(
                        batch_size,
                        lc.num_constraints(),
                        self.vector_structure.size,
                    )
                )
                if lc.M is not None:
                    row = row.at[:, :, lc.idxs_w].set(-lc.M)
                else:
                    row = row.at[:, :, lc.idxs_w].set(
                        -jnp.eye(simple_slice_len(lc.idxs_w))
                    )
                row = row.at[:, :, self.vector_structure[f"w_{i}"]].set(
                    jnp.expand_dims(jnp.eye(lc.num_constraints()), axis=0)
                )
                row = jnp.tile(row, (mbAC // batch_size, 1, 1))
                assert row.shape[-1] == self.vector_structure.size
                As.append(row)
                bs.append(lc.q)
        A_lifted = jnp.concatenate([first_row_batched, *As], axis=1)
        b_lifted = jnp.concatenate([self.eq_constraint.b, *bs], axis=1)
        eq_lifted = EqualityConstraint(
            A=A_lifted,
            b=b_lifted,
            method=method,
            var_b=self.eq_constraint.var_b,
            var_A=self.eq_constraint.var_A,
        )

        if self.ineq_constraint is None:
            box_lifted = self.box_constraint
        elif self.box_constraint is None:
            # We only project the lifted part.
            box_mask = np.concatenate(
                [np.zeros(self.dim, dtype=bool), np.ones(self.n_ineq, dtype=bool)]
            )
            box_lifted = BoxConstraint(
                BoxConstraintSpecification(
                    lb=self.ineq_constraint.lb,
                    ub=self.ineq_constraint.ub,
                    mask=box_mask,
                )
            )
        else:
            # We project both the lifted and the initial box
            box_mask = jnp.concatenate(
                [
                    self.box_constraint.mask,
                    jnp.ones(self.n_ineq, dtype=bool),
                ]
            )
            # Maximum batch dimension for lower bound
            mblb = max(
                self.box_constraint.lb.shape[0],
                self.ineq_constraint.lb.shape[0],
            )
            lifted_lb = jnp.concatenate(
                [
                    jnp.tile(
                        self.box_constraint.lb,
                        (mblb // self.box_constraint.lb.shape[0], 1, 1),
                    ),
                    jnp.tile(
                        self.ineq_constraint.lb,
                        (mblb // self.ineq_constraint.lb.shape[0], 1, 1),
                    ),
                ],
                axis=1,
            )
            # Maximum batch dimension for upper bound
            mbub = max(
                self.box_constraint.ub.shape[0],
                self.ineq_constraint.ub.shape[0],
            )
            lifted_ub = jnp.concatenate(
                [
                    jnp.tile(
                        self.box_constraint.ub,
                        (mbub // self.box_constraint.ub.shape[0], 1, 1),
                    ),
                    jnp.tile(
                        self.ineq_constraint.ub,
                        (mbub // self.ineq_constraint.ub.shape[0], 1, 1),
                    ),
                ],
                axis=1,
            )
            box_lifted = BoxConstraint(
                BoxConstraintSpecification(
                    lb=lifted_lb,
                    ub=lifted_ub,
                    mask=box_mask,
                )
            )

        cross_lifted = [
            CrossConstraint(
                idxs_z=lc.idxs_z,
                idxs_w=self.vector_structure[f"w_{i}"],
                M=None,
                q=None,
                dim=self.dim,
            )
            for i, lc in enumerate(self.nl_constraints)
        ]

        cartesian = CartesianConstraint(box_lifted, cross_lifted)

        def lift(y):
            """Lift the input to the lifted dimension."""
            xs = [y.x]
            if self.ineq_constraint:
                xs.append(self.ineq_constraint.C @ y.x)
            for nl in self.nl_constraints:
                xs.append(nl.get_auxiliary_variables(y.x))
            y = y.update(x=jnp.concatenate(xs, axis=1))
            if self.eq_constraint.var_b:
                bs_batch = [
                    jnp.tile(b, (y.eq.b.shape[0] // b.shape[0], 1, 1)) for b in bs
                ]
                y = y.update(
                    eq=y.eq.update(
                        b=jnp.concatenate(
                            [
                                y.eq.b,
                                *bs_batch,
                            ],
                            axis=1,
                        )
                    )
                )
            if self.eq_constraint.var_A:
                As_batch = [
                    jnp.tile(A, (y.eq.A.shape[0] // A.shape[0], 1, 1)) for A in As
                ]
                y = y.update(
                    eq=y.eq.update(
                        A=jnp.concatenate(
                            [
                                y.eq.A,
                                *As_batch,
                            ],
                            axis=1,
                        )
                    )
                )

            return y

        return (eq_lifted, cartesian, lift)
