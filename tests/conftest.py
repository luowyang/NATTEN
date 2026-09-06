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
