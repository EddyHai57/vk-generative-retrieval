"""G6-PREP mutation helper: re-execute a module's source with one textual change and patch its public objects.

``mutated(module, old, new)`` asserts that ``old`` occurs exactly once (so a mutant can never be vacuous), executes the
changed source in a fresh namespace and, for the duration of the ``with`` block, replaces every function and class the
module defines by its mutated twin. Tests must therefore reach the module's objects through the module attribute
(``sd.name``) at call time, never through ``from module import name``. Not a test module.
"""
import contextlib
import inspect
import io
import unittest
from unittest import mock


def run_ids(ids):
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in ids)
    return unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)


@contextlib.contextmanager
def mutated(module, old, new, count=1):
    source = inspect.getsource(module)
    found = source.count(old)
    if found != count:
        raise AssertionError(f'mutation anchor occurs {found} times, expected {count}: {old!r}')
    namespace = {'__name__': module.__name__, '__package__': module.__package__, '__file__': module.__file__}
    exec(compile(source.replace(old, new), module.__file__, 'exec'), namespace)
    patches = {key: value for key, value in namespace.items()
               if (inspect.isfunction(value) or inspect.isclass(value)) and getattr(value, '__module__', None) == module.__name__}
    patches.update({key: value for key, value in namespace.items()
                    if key.isupper() and key in vars(module) and not callable(value)})
    with mock.patch.multiple(module, **patches):
        yield


def assert_killed(case, module, old, new, test_ids):
    """The selected tests must fail (or error) under the mutant."""
    with mutated(module, old, new):
        result = run_ids(test_ids)
    case.assertFalse(result.wasSuccessful(), f'mutant survived: {old!r} -> {new!r}')
    return result
