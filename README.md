![](https://twemoji.maxcdn.com/v/latest/72x72/1f938.png)

# FLEXible

[![Tests](https://github.com/FLEXible-FL/FLEX-framework/actions/workflows/pytest.yml/badge.svg)](https://github.com/FLEXible-FL/FLEX-framework/actions/workflows/pytest.yml)
[![Linter](https://github.com/FLEXible-FL/FLEX-framework/actions/workflows/trunk.yml/badge.svg)](https://github.com/FLEXible-FL/FLEX-framework/actions/workflows/trunk.yml)

The documentation of the package was generated using [pdoc](https://pdoc3.github.io/pdoc/). The steps to generate the documentation are the following ones:
- mkdir docs
- pdoc flex -o=docs --docformat markdown --footer-text "FLEXible: Federad Learning Experiments"

In order to install this repo locally, to develop:

``
    pip install -e ".[develop]"
``

With only tensorflow support:

``
    pip install -e ".[tf]"
``

With only pytorch support:

``
    pip install -e ".[pt]"
``

Without support for any particular framework

``
    pip install -e .
``
