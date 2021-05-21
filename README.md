# `digitize-ecg`

A library and command-line tool for digitizing electrocardiograms images


## Getting Started

If you have already followed the `paper-ecg` setup guide, just set `pyenv local paper-ecg` in the repository root directory and you are good to go.

This assumes a working version of [PyEnv](https://github.com/pyenv/pyenv#readme).


### Install Python 3.6.7

Install Python 3.6.7 using `pyenv install`:

```bash
pyenv install 3.6.7
```


### Set up an environment

1. Use `pyenv` to create a virtual environment for the project. 

    ```bash
    pyenv virtualenv 3.6.7 digitize-ecg
    ```

2. Navigate to the project root directory (`.../paper-ecg/`) and assign the virtual environment you just created to the current directory (this automatically activates the environment).

    ```bash
    pyenv local paper-ecg
    ```

    Now, whenever this folder is the working directory in terminal, this environment will be used.
    Test this out by running:

    ```bash
    > python --version
    3.6.7
    ```


### Install dependencies

Use `pip install` to install required dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ This may take several minutes


### Uninstalling

If you no longer want to have the `paper-ecg` virtualenv, you can delete it:

```bash
pyenv virtualenv-delete paper-ecg
```

If you wish to remove Python 3.6.7:
```bash
pyenv uninstall 3.6.7
```


## Dependencies

The project currently requires Python `3.6.7` to work with `paper-ecg`, which depends on `fbs` (see [3.7 support issue](https://github.com/mherrmann/fbs/issues/61)).
