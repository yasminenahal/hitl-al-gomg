#!/usr/bin/env python
#  coding=utf-8
import pytest
import argparse


TESTS_FOLDER = 'unittest_reinvent'


parser = argparse.ArgumentParser(description='Run reinvent_scoring tests')
parser.add_argument(
    '--unittests', action='store_true',
    help='Only run unittests (Please indicate either integration or unittests flag)'
)
parser.add_argument(
    '--integration', action='store_true',
    help='Only run integration tests (Please indicate either integration or unittests flag)'
)

args, _ = parser.parse_known_args()


if args.unittests:
    pytest_args = ['-m', 'not integration', TESTS_FOLDER]
elif args.integration:
    pytest_args = ['-m', 'integration', TESTS_FOLDER]
else:
    raise Exception('Please provide either --unittests or --integration flag.')


if __name__ == '__main__':
    pytest.main(pytest_args)
