# Final Project: Synthesis of Mechanisms
**Austin Wright and Rohit Ramesh**

## Install

Detailed installation instructions are at [readme/INSTALL.md](readme/INSTALL.md). 

## Project Overview


## Style Guide and Documentation
 
We will be trying to keep everything here as functional as possible, 
therefore avoid global state as much as feasible.
We also will be using static typechecking as much as python supports, 
using the [typing](https://docs.python.org/3/library/typing.html#) package.
In addition code should include documentation in alignemnt with python 
standard [docstrings](https://www.python.org/dev/peps/pep-0257/).

### Typechecking 

The following command typechecks the entire system. 

   $ mypy mechsynth --ignore-missing-imports
 
We liberally use `# type: ignore` to suppress stupid type errors. 
Just make sure you only add those for type errors that only exist due to 
limitations with python's type system. 
If you can actually fix a type error, do that.

**NOTE:** I'm doing enough hackery with decorators and other similar 
meta-programming constructs that the type-checker is basically useless.

### Tests 

Run tests with the following command. 

   $ pytest -v --cache-clear 

Add tests to `mechsynth\test`. 

## Repository Structure

```tree
.
├── mechsynth
│   ├── errors.py
│   ├── functor.py
│   ├── id.py
│   └── test
│       ├── test_functor.py
│       └── test_id.py
├── readme
│   └── INSTALL.md
├── README.m 
├── requirements.txt
├── setup
│   ├── dreal-test.py
│   ├── init.sh
│   ├── pybind11.patch
│   └── workspace.patch
├── setup.py
├── TODO.md
└── Vagrantfile

```

TODO :: Add notes to the list of important files above.ppp p

We basically just follow the standard python setup here. 
