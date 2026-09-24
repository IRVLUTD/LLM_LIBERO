"""Optional per-task hints distilled from watching one human demonstration.

These are NOT zero-shot: they encode demonstration knowledge. The ``libero``
embodiment appends them to its operating notes only with ``task_notes=True``.
Keys are native LIBERO-10 task ids.
"""

from __future__ import annotations

TASK_NOTES: dict[int, str] = {
    2: 'Task-specific note from watching one successful human demonstration of '
       'this task: approach the stove control from directly above and grasp the '
       'knob itself, not the burner plate around it. Turning it on is a ROTATION '
       'in place while gripping, not a translation -- release only once the '
       'burner visibly glows red in the image; do not keep rotating past that '
       'point. Separately, based on prior runs of this task where the model was '
       'shown the actual demonstration images directly (higher fidelity than a '
       'text summary can capture): the moka pot should be grasped by its handle '
       '(the protrusion to one side), not the broad body or the lid -- grasping '
       'the body risks tipping the pot over. Carry the pot directly above the '
       'now-lit burner before lowering and releasing it.',
    3: 'Task-specific note from watching one successful human demonstration of '
       'this task: grasp the bowl by its rim from directly above, not from the '
       'side or by reaching into it. Once grasped, lift well clear of the '
       'tabletop and cabinet top BEFORE moving laterally toward the open '
       'drawer -- the successful demo moves the bowl high and clear of the '
       'cabinet face, not a low sweep past it, then descends straight down '
       'only once positioned over the drawer opening. Release the bowl inside '
       'the drawer, then retract the arm and approach the drawer front '
       'separately to push it closed -- closing is a distinct final motion '
       'after releasing the bowl, not done while still holding it.',
}
