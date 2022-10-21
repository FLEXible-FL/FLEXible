from setuptools import find_packages, setup

_deps = ["numpy"]

install_requires = _deps

setup(
    name="flex",
    version="0.0.1.dev0",
    author="The FLEXible team",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="FL federated-learning flexible",
    url="https://github.com/FLEXible-FL/FLEX-framework",
    packages=find_packages(),
    install_requires=["numpy",
                    "multiprocess",
                    "sklearn",
                    "cardinality",
                    "lazyarray"
                    ],
    extras_require={
        "tensorflow": ["tensorflow", 
                "tensorflow_datasets", 
                "tensorflow_hub"
                ],
        "pytorch": ["torch", 
                "torchvision", 
                "torchtext", 
                "torchdata"
                ],
        "hugginface": ["datasets"],
        "all": ["pytest",
                "pytest-cov",
                "jinja2",
                "tensorflow",
                "tensorflow_datasets",
                "tensorflow_hub",
                "torch",
                "torchvision",
                "torchtext",
                "torchdata",
                "datasets"
                ],
    },
    python_requires=">=3.9.0",
)
