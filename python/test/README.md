# Dependency
`pytest`
`pytest-asyncio`
`pytest-repeat`

# Run Command
- run all the tests in a module: `pytest`
- run all the tests in a file: `pytest -v -s test_example.py`
- run specific test: `pytest -k test_addition_negative`
- only run async tests: `pytest -k asyncio`
- run test for multiple times: `pytest --count <n>`