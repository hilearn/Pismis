## Getting Started

First install system dependencies, e.g. python3.5
```
sudo ./setup.sh
```

Run `make update_venv` to install the latest requirements
(it will do nothing if you rerun without changeing dependencies)
```
make update_venv
```

Then activate the created virutalenv
```
source .venv/current/bin/activate
```

## Running tests

We run tests nose

```
make test
```
