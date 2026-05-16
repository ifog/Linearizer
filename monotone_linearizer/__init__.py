"""Monotone Linearizer: expressivity-preserving equilibrium models via learned coordinates.

Replaces the linear core A of the original Linearizer with a maximally monotone
operator M, so the model is a DEQ with guaranteed unique fixed point and
provably-convergent iteration, while a learned invertible flow g lifts
monDEQ's expressivity restriction by transporting the monotonicity constraint
into a learned coordinate system.
"""
