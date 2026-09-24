"""LIBERO-10 as an Inspect Robots task: one scene per LIBERO initial state.

Importing LIBERO is slow and can prompt interactively on first use, so task
names and instructions are a static copy of
``benchmark.get_benchmark_dict()["libero_10"]()`` (native task order).
"""

from __future__ import annotations

from inspect_robots.errors import ConfigError
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Task

# Native LIBERO-10 task id -> (BDDL task name, language instruction).
LIBERO_10_TASKS: tuple[tuple[str, str], ...] = (
    (
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        "put both the alphabet soup and the tomato sauce in the basket",
    ),
    (
        "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
        "put both the cream cheese box and the butter in the basket",
    ),
    (
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        "turn on the stove and put the moka pot on it",
    ),
    (
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
        "put the black bowl in the bottom drawer of the cabinet and close it",
    ),
    (
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        "put the white mug on the left plate and put the yellow and white mug on the right plate",
    ),
    (
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        "pick up the book and place it in the back compartment of the caddy",
    ),
    (
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
        "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    ),
    (
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        "put both the alphabet soup and the cream cheese box in the basket",
    ),
    (
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
        "put both moka pots on the stove",
    ),
    (
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
        "put the yellow and white mug in the microwave and close it",
    ),
)

NUM_INIT_STATES = 50
MAX_STEPS = 900
# LIBERO env seed for init state i is ENV_SEED_BASE + i (matches the X-VLA
# LIBERO client: 42 + episode + 100).
ENV_SEED_BASE = 142


def check_task_id(task_id: int) -> int:
    """Return ``task_id`` if it names a LIBERO-10 task, else raise ``ConfigError``."""
    if not isinstance(task_id, int) or not 0 <= task_id < len(LIBERO_10_TASKS):
        raise ConfigError(
            f"task_id must be an integer in 0..{len(LIBERO_10_TASKS) - 1}, got {task_id!r}.\n"
            "fix: pass -T task_id=N (native LIBERO-10 order, see README)"
        )
    return task_id


def libero_10_task(
    task_id: int = 0,
    num_scenes: int = 1,
    start: int = 0,
    max_steps: int = MAX_STEPS,
) -> Task:
    """Build one LIBERO-10 task over init states ``start .. start+num_scenes-1``.

    Each scene carries ``libero_task_id`` and ``init_state`` in its metadata; the
    ``libero`` embodiment reads them on reset, so ``-E task_id`` is optional.
    Success is the LIBERO environment predicate at the end of the trial.
    """
    check_task_id(task_id)
    if not 0 <= start < start + num_scenes <= NUM_INIT_STATES:
        raise ConfigError(
            f"init states {start}..{start + num_scenes - 1} are outside 0..{NUM_INIT_STATES - 1}.\n"
            "fix: choose -T start and -T num_scenes within LIBERO's 50 initial states"
        )
    name, instruction = LIBERO_10_TASKS[task_id]
    scenes = [
        Scene(
            id=f"task{task_id:02d}_init{i:02d}",
            instruction=instruction,
            init_seed=ENV_SEED_BASE + i,
            metadata={"libero_task_id": task_id, "init_state": i, "libero_task_name": name},
        )
        for i in range(start, start + num_scenes)
    ]
    return Task(
        name=f"libero_10_task{task_id:02d}",
        scenes=scenes,
        scorer=success_at_end(),
        max_steps=max_steps,
    )
