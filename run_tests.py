"""
Automated test runner script.
"""
import sys
import pytest

if __name__ == "__main__":
    ret = pytest.main(["tests", "-v", "-s", "--color=no"])
    sys.exit(ret)
