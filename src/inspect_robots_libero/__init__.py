"""inspect-robots-libero: LIBERO-10 embodiment and tasks for Inspect Robots.

Install next to ``inspect-robots`` and ``inspect-robots-agent`` and the
``libero`` embodiment plus the ``libero_10`` task appear in the registry, so any
registered policy (including LLMs through ``--policy agent``) can run them::

    inspect-robots list embodiments        # -> includes "libero"
    inspect-robots run --task libero_10 -T task_id=2 --embodiment libero \
        --policy agent -P model=anthropic/claude-opus-5-5 -P wire=messages

or programmatically::

    from inspect_robots import eval
    from inspect_robots_agent.policy import agent_policy
    from inspect_robots_libero import libero_10, libero_embodiment
    eval(libero_10(task_id=2), agent_policy(model="openai/gpt-6-sol"), libero_embodiment())

Both are discovered through entry points, so neither needs importing first.
"""

from __future__ import annotations

from typing import Any

from inspect_robots.task import Task

from inspect_robots_libero.embodiment import LiberoEmbodiment
from inspect_robots_libero.tasks import LIBERO_10_TASKS, libero_10_task

__all__ = ["LIBERO_10_TASKS", "LiberoEmbodiment", "libero_10", "libero_embodiment"]

__version__ = "0.1.0"


def libero_embodiment(**kwargs: Any) -> LiberoEmbodiment:
    """Factory the registry calls (entry point ``libero``); the CLI forwards ``-E k=v``."""
    return LiberoEmbodiment(**kwargs)


def libero_10(**kwargs: Any) -> Task:
    """Factory the registry calls (entry point ``libero_10``); the CLI forwards ``-T k=v``."""
    return libero_10_task(**kwargs)
