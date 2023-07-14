![](https://twemoji.maxcdn.com/v/latest/72x72/1f938.png)

# FLEXible

[![Tests](https://github.com/FLEXible-FL/FLEX-framework/actions/workflows/pytest.yml/badge.svg)](https://github.com/FLEXible-FL/FLEX-framework/actions/workflows/pytest.yml)
[![Linter](https://github.com/FLEXible-FL/FLEX-framework/actions/workflows/trunk.yml/badge.svg)](https://github.com/FLEXible-FL/FLEX-framework/actions/workflows/trunk.yml)

The documentation of the package was generated using [pdoc](https://pdoc3.github.io/pdoc/). The steps to generate the documentation are the following ones:

``
pdoc flex -o=docs --docformat markdown --footer-text "FLEXible: Federad Learning Experiments"
``

In order to install this repo locally for development:

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

### Others
**Authors order**: The order of the authors was chosen by asking ChatGPT to make a program to do it.

The conversation was:

Q: "Escribe en Python un programa que elija entre dos personas aleatoriamente, Alberto y Daniel, el orden entre estos dada una semilla aleatoria que tú elijas. Además, ejecuta el programa y dime el resultado."

A: "
Aquí tienes un programa en Python que selecciona aleatoriamente el orden entre dos personas, Alberto y Daniel, utilizando una semilla aleatoria predefinida.

```python
import random

# Semilla aleatoria predefinida
random.seed(123)

# Lista con los nombres de las personas
personas = ["Alberto", "Daniel"]

# Elección aleatoria del orden
orden = random.sample(personas, k=2)

# Imprimir el resultado
print("El orden seleccionado es:", orden)
```

Al ejecutar este programa, el resultado puede variar dependiendo de la semilla aleatoria seleccionada. Sin embargo, dado que hemos establecido una semilla específica (`random.seed(123)`), el resultado será el mismo cada vez que se ejecute con esa semilla.

Ejemplo de resultado:

```
El orden seleccionado es: ['Daniel', 'Alberto']
```

Recuerda que si cambias la semilla aleatoria o ejecutas el programa en otro momento, el resultado será diferente."
