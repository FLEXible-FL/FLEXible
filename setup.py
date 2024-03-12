"""
Copyright (C) 2024  Instituto Andaluz Interuniversitario en Ciencia de Datos e Inteligencia Computacional (DaSCI)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from setuptools import find_packages, setup


TF_requires = ["tensorflow<2.11", # https://github.com/tensorflow/tensorflow/issues/58973
                "tensorflow_datasets", 
                "tensorflow_hub"
        ]
PT_requires = ["torch", 
                "torchvision", 
                "torchtext", 
                "torchdata",
                "portalocker",
        ]
HF_requires = ["datasets"]

setup(
        name="flexible-fl",
        version="0.6.1",
        author="Jimenez-Lopez Daniel, Argente-Garrido Alberto",
        author_email="xehartnort@gmail.com, albertoargentedcgarrido@gmail.com",
        long_description=open("README.md", "r", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        keywords="FL federated-learning flexible",
        url="https://github.com/FLEXible-FL/FLEX-framework",
        packages=find_packages(),
        install_requires=["numpy",
                        "multiprocess",
                        "scikit-learn",
                        "cardinality",
                        "sultan",
                        "tqdm",
                        "scipy",
                        "gdown",
                        "tensorly"
                        ],
        extras_require={
                "tensorflow": TF_requires,
                "pytorch": PT_requires,
                "hugginface": HF_requires,
                "develop": ["pytest",
                        "pytest-cov",
                        "pytest-xdist",
                        "coverage",
                        "jinja2",
                        *TF_requires,
                        *PT_requires,
                        *HF_requires
                        ],
        },
        python_requires=">=3.8.10",
        classifiers=[
                "Development Status :: 4 - Beta",
                "Intended Audience :: Science/Research",
                "Topic :: Software Development :: Build Tools",
                # Pick your license as you wish
                "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
                # Specify the Python versions you support here. In particular, ensure
                # that you indicate you support Python 3. These classifiers are *not*
                # checked by 'pip install'. See instead 'python_requires' below.
                "Programming Language :: Python :: 3",
                "Programming Language :: Python :: 3.8",
                "Programming Language :: Python :: 3.9",
                "Programming Language :: Python :: 3.10",
                "Programming Language :: Python :: 3.11",
                "Programming Language :: Python :: 3 :: Only",
        ],
)
