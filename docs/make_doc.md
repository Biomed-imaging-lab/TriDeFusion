# Makefile and Make command documentation

## Introduction

`make` is a build automation tool used to compile and manage projects efficiently. It automates execution of commands defined in a Makefile, a script-like configuration file that specifies dependencies and build rules.

## Installation

### Linux (Ubuntu/Debian)

```bash
sudo apt-get install make
```

### macOS (Homebrew)

```bash
brew install make
```

### Windows (Chocolatey)

```bash
choco install make
```

## Makefile

A Makefile consists of:

- Targets - The names of tasks to execute.
- Dependencies - Files required to execute the target.
- Commands - Shell commands executed when the target is invoked.

### Makefile Example

```bash
make <target>
```

### Runing `make` commands

1. Install dependencies

    ```bash
    make install
    ```

2. Run all tests

    ```bash
    make tests
    ```

3. Run specific test

    ```bash
    make test <test_name>
    ```

4. Deploy docker container

    ```bash
    make deploy
    ```

5. Run inference

    ```bash
    make inference
    ```
