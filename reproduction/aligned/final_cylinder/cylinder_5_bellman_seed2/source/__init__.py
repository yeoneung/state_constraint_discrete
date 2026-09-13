"""SICON revision: one monotone operator for evaluation and improvement."""

from .problem import Problem, RunConfig
from .operators import differences, policy_action, residual, stencil_audit

__all__ = ["Problem", "RunConfig", "differences", "policy_action", "residual", "stencil_audit"]
