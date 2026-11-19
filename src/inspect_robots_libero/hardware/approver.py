"""Keep the framework limiter and whole-path hardware preflight in one frame."""
import json

import numpy as np
from inspect_robots.approver import ClampApprover, DeltaLimitApprover
from inspect_robots.errors import SafetyAbort
from inspect_robots.types import Action


class HardwarePathApprover:
    """Rebase once per new path, retaining all per-waypoint and motor limits.

    A new tool path starts at the policy's measured observation. That can differ
    from the last requested command due to physical tracking error. Reusing the
    old command as the limiter reference silently changes preflighted targets.
    References are frozen per accepted path, never refreshed on every tick.
    """
    def __init__(self, embodiment):
        self.embodiment = embodiment
        self.bounds = ClampApprover(embodiment.info.action_space)
        self.delta = DeltaLimitApprover(embodiment.info.action_space)

    def review(self, action, store):
        """Approve the exact checked target, or stop instead of sending a rewritten path."""
        if action.meta.get("request_stop"):
            return action  # The adapter handles stop as a hold, not a pose command.
        context = self.embodiment.approval_context()
        if context is None:
            raise SafetyAbort("Hardware action has no active whole-path preflight")
        generation, reference = context
        reference = np.asarray(reference, dtype=float)
        if reference.shape != self.embodiment.info.action_space.shape or not np.isfinite(reference).all():
            raise SafetyAbort("Hardware preflight has an invalid observation reference")
        key = "_llm_libero_hardware_preflight_generation"
        if store.get(key) != generation:
            DeltaLimitApprover.rewind_reference(store, reference)
            self.delta.review(Action(data=reference), store)  # Initialize a fresh trial store too.
            store[key] = generation
            with (self.embodiment.output/"approver_events.jsonl").open("a") as file:
                file.write(json.dumps(dict(event="reference_rebased_for_preflight", generation=generation,
                                           reference=reference.tolist()))+"\n")
        bounded = self.bounds.review(action, store)
        reviewed = self.delta.review(bounded, store)
        if not np.array_equal(reviewed.data, action.data):
            raise SafetyAbort("Hardware waypoint would be rewritten after preflight; no modified target was sent")
        return action
