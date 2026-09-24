"""Task factory and entry-point wiring; no LIBERO or API needed."""
import os
import sys
import unittest
from unittest import mock

from inspect_robots.errors import ConfigError
from inspect_robots.registry import registered

from inspect_robots_libero import LIBERO_10_TASKS, LiberoEmbodiment, libero_10


class TaskTests(unittest.TestCase):
    def test_scenes_carry_task_and_init_state(self):
        task = libero_10(task_id=2, num_scenes=3, start=5)
        self.assertEqual([s.metadata['init_state'] for s in task.scenes], [5, 6, 7])
        self.assertEqual({s.metadata['libero_task_id'] for s in task.scenes}, {2})
        self.assertEqual([s.init_seed for s in task.scenes], [147, 148, 149])
        self.assertEqual(task.scenes[0].instruction, LIBERO_10_TASKS[2][1])

    def test_rejects_bad_ids(self):
        with self.assertRaises(ConfigError):
            libero_10(task_id=10)
        with self.assertRaises(ConfigError):
            libero_10(task_id=0, num_scenes=2, start=49)

    def test_entry_points_registered(self):
        self.assertIn('libero', registered('embodiment'))
        self.assertIn('libero_10', registered('task'))
        self.assertIn('agent', registered('policy'))

    def test_embodiment_defaults_to_current_interpreter(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('LIBERO_PYTHON', None)
            self.assertEqual(LiberoEmbodiment().libero_python, sys.executable)

    def test_task_notes_need_task_id(self):
        with self.assertRaises(ConfigError):
            LiberoEmbodiment(libero_python='/bin/true', task_notes=True)


if __name__ == '__main__':
    unittest.main()
