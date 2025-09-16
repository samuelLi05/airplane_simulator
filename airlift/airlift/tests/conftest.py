import pytest
from airlift.utils import definitions

definitions.TEST_MODE = True

def pytest_addoption(parser):
    parser.addoption("--render", action="count", default=0)
    #parser.addoption("--redis-exec", action="store", default="redis-server")

@pytest.fixture
def render(request):
    return request.config.getoption("--render") > 0
