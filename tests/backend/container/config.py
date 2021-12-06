import os

from autograde.util import parse_bool

SKIP_CONTAINER = parse_bool(os.getenv('AG_TEST_SKIP_CONTAINER', False))
BUILD_IMAGE_FROM_SOURCE = parse_bool(os.getenv('AG_TEST_BUILD_IMAGE_FROM_SOURCE', False))
