import pytest
import torch
import torch._dynamo

from .utils import logger

DYNAMO_BUDGET = (
    "cache_size_limit",
    "recompile_limit",
    "accumulated_recompile_limit",
    "fail_on_recompile_limit_hit",
)


@pytest.fixture(autouse=True)
def log_test_name(request):
    logger.debug(f"Starting {request.node.name}")
    yield
    logger.debug(f"Finished {request.node.name}")


@pytest.fixture(autouse=True)
def dynamo_recompile_budget():
    """Keep one test's compile budget out of the next test.

    torch._dynamo's recompile and cache-size limits are process-global, and the
    modules that tighten them do so for their own compiled callables and leave them
    tightened. A later test that compiles anything then runs under a budget it was
    never written for, which is why those modules pass alone and fail in a
    whole-suite run.
    """
    saved = {name: getattr(torch._dynamo.config, name) for name in DYNAMO_BUDGET}
    yield
    for name, value in saved.items():
        setattr(torch._dynamo.config, name, value)


@pytest.fixture(autouse=True)
def default_device():
    """Keep one test's default device out of the next test.

    Eight modules switch the process-global default device to cuda so their own
    unqualified tensors land there, and leave it switched. A later test that builds
    a generator or a tensor without naming a device then inherits cuda from whoever
    ran first, and fails on the mismatch with the ones it does name.

    Clearing first is what keeps this fixture from leaving global state of its own:
    torch.set_default_device installs a mode that every subsequent torch call is
    dispatched through, and passing it the cpu that get_default_device reports for a
    session which never set a default device would install one where there was none.
    set_default_device(None) is that session's actual state, and the restore below it
    runs only for a test that inherited a default device to go back to.
    """
    saved = torch.get_default_device()
    yield
    torch.set_default_device(None)
    if torch.get_default_device() != saved:
        torch.set_default_device(saved)
