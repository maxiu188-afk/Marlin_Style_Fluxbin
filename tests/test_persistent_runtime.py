"""Compatibility key changes must prevent accidental reuse of incompatible venvs."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('persistent_runtime',
    Path(__file__).resolve().parents[1]/'scripts/prepare_persistent_runtime.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PersistentRuntimeTests(unittest.TestCase):
    def test_order_independent(self):
        self.assertEqual(module.fingerprint({'python':'3.12','arch':'8.0'}),
                         module.fingerprint({'arch':'8.0','python':'3.12'}))

    def test_compatibility_changes_invalidate_key(self):
        base = {'python':'3.12','machine':'x86_64','torch':'2.8','lock':'a','repo':'a'}
        for field in base:
            with self.subTest(field=field):
                self.assertNotEqual(module.fingerprint(base),
                                    module.fingerprint(dict(base, **{field:'changed'})))


if __name__ == '__main__':
    unittest.main()
