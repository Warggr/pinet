"""Parser of constraints to lifted representation module."""

from typing import Optional, Sequence

import jax.numpy as jnp
import numpy as np

from pinet.dataclasses import BoxConstraintSpecification

from .affine_equality import EqualityConstraint
from .affine_inequality import AffineInequalityConstraint
from .box import BoxConstraint
from .cross import CrossConstraint


class VectorStructure:
    """Helper class to describe block vectors and block matrices."""

    def __init__(self, sizes: Sequence[tuple[str, int]]):
        """Constructor."""
        cuts = {}
        start = 0
        for k, v in sizes:
            cuts[k] = slice(start, start + v)
            start += v
        self.size = start
        self.cuts = cuts

    def __getitem__(self, idx: str | Sequence[str] | slice) -> slice:
        """Get item."""
        if isinstance(idx, str):
            return self.cuts[idx]
        elif isinstance(idx, slice):
            if idx.step is not None:
                raise ValueError("slice step [start:stop:step] not supported")
            start, stop = None, None
            if idx.start is None:
                start = 0
            if idx.stop is None:
                stop = self.size
            for name, cut in self.cuts.items():
                if name == idx.start:
                    start = cut.start
                if name == idx.stop:
                    stop = cut.start
                    break
            if start is None or stop is None:
                raise ValueError(f"{idx.start} or {idx.stop} not found")
            return slice(start, stop)
        else:
            first, *more = idx
            start = self.cuts[first].start
            stop = self.cuts[first].stop
            for idx in more:
                next_cut = self.cuts[idx]
                assert next_cut.start == stop
                stop = next_cut.stop
            return slice(start, stop)


class ConstraintParser:
    """Parse constraints into a lifted representation.

    This class takes as input an equality, an inequality, and a box constraint.
    It returns an equivalent equality and box constraint in a lifted representation.
    """

    def __init__(
        self,
        eq_constraint: EqualityConstraint | None,
        ineq_constraint: AffineInequalityConstraint | None,
        cross_constraint: CrossConstraint | None = None,
        box_constraint: BoxConstraint | None = None,
    ) -> None:
        """Initiaze the constraint parser.

        Args:
            eq_constraint (EqualityConstraint): An equality constraint.
            ineq_constraint (AffineInequalityConstraint): An inequality constraint.
            box_constraint (BoxConstraint): A box constraint.
            cross_constraint (CrossConstraint): A cross constraint.
        """
        if ineq_constraint is None and cross_constraint is None:
            # The constraints do not need lifting.
            self.parse = lambda method: (eq_constraint, box_constraint, lambda y: y)
            return

        self.dim = (
            ineq_constraint.dim if ineq_constraint is not None else cross_constraint.dim
        )
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
        self.n_ineq = ineq_constraint.n_constraints if ineq_constraint else 0
        self.box_constraint = box_constraint
        self.cross_constraint = cross_constraint
        self.n_cross = cross_constraint.n_constraints if cross_constraint else 0

        vector_structure = [("y", self.dim)]
        if ineq_constraint is not None:
            vector_structure.append(("y_aux", self.n_ineq))
        if cross_constraint is not None:
            vector_structure += [
                ("w", self.cross_constraint.dim),
                ("z'", self.cross_constraint.dim),
                ("w'", self.cross_constraint.dim),
            ]
        self.vector_structure = VectorStructure(vector_structure)

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
    ) -> tuple[EqualityConstraint, BoxConstraint, CrossConstraint]:
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
        rows, bs = [], []
        first_row_batched = jnp.tile(
            jnp.concatenate(
                [
                    self.eq_constraint.A,
                    jnp.zeros(
                        shape=(
                            self.eq_constraint.A.shape[0],
                            self.n_eq,
                            self.n_ineq + self.n_cross,
                        )
                    ),
                ],
                axis=2,
            ),
            (mbAC // self.eq_constraint.A.shape[0], 1, 1),
        )
        rows.append(first_row_batched)
        bs.append(self.eq_constraint.b)

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
                                self.n_cross,
                            )
                        ),
                    ],
                    axis=2,
                ),
                (mbAC // self.ineq_constraint.C.shape[0], 1, 1),
            )
            rows.append(second_row_batched)
            bs.append(jnp.zeros(shape=(self.eq_constraint.b.shape[0], self.n_ineq, 1)))
        if self.cross_constraint:
            row = jnp.zeros(
                shape=(
                    self.cross_constraint.M.shape[0],
                    self.cross_constraint.M.shape[1],
                    self.vector_structure.size,
                )
            )
            row = row.at[:, :, self.cross_constraint.idxs_z].set(
                -self.cross_constraint.M
            )
            row = row.at[:, :, self.vector_structure["w"]].set(
                jnp.expand_dims(jnp.eye(self.n_cross), axis=0)
            )
            row = jnp.tile(row, (mbAC // self.cross_constraint.M.shape[0], 1, 1))
            rows.append(row)
            bs.append(self.cross_constraint.q)

            row = jnp.zeros(
                shape=(self.cross_constraint.M.shape[1], self.vector_structure.size)
            )
            row = row.at[:, :, self.cross_constraint.idxs_z].set(-jnp.eye(self.n_cross))
            row = row.at[:, :, self.vector_structure["z'"]].set(jnp.eye(self.n_cross))
            row = jnp.expand_dims(row, axis=0)
            row = jnp.tile(row, (mbAC // self.cross_constraint.M.shape[0], 1, 1))
            rows.append(row)
            bs.append(jnp.zeros(1, self.n_cross, 1))

            row = jnp.zeros(
                shape=(self.cross_constraint.M.shape[1], self.vector_structure.size)
            )
            row = row.at[:, :, self.vector_structure["w"]].set(-jnp.eye(self.n_cross))
            row = row.at[:, :, self.vector_structure["w'"]].set(jnp.eye(self.n_cross))
            row = jnp.expand_dims(row, axis=0)
            row = jnp.tile(row, (mbAC // self.cross_constraint.M.shape[0], 1, 1))
            rows.append(row)
            bs.append(jnp.zeros(1, self.n_cross, 1))
        A_lifted = jnp.concatenate(rows, axis=1)
        b_lifted = jnp.concatenate(bs, axis=1)
        eq_lifted = EqualityConstraint(
            A=A_lifted,
            b=b_lifted,
            method=method,
            var_b=self.eq_constraint.var_b,
            var_A=self.eq_constraint.var_A,
        )

        if self.box_constraint is None:
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

        def lift(y):
            """Lift the input to the lifted dimension."""
            xs = [y.x]
            if self.ineq_constraint:
                xs.append(self.ineq_constraint.C @ y.x)
            if self.cross_constraint:
                z = y.x[self.cross_constraint.idxs_z]
                w = self.cross_constraint.M @ z + self.cross_constraint.q
                xs += [w, z, w]
            y = y.update(x=jnp.concatenate(xs, axis=1))
            if self.eq_constraint.var_b:
                y = y.update(
                    eq=y.eq.update(
                        b=jnp.concatenate(
                            [
                                y.eq.b,
                                jnp.zeros(
                                    (y.x.shape[0], self.n_ineq + 3 * self.n_cross, 1)
                                ),
                            ],
                            axis=1,
                        )
                    )
                )
            return y

        return (eq_lifted, box_lifted, lift)
