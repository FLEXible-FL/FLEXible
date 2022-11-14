from setuptools import find_packages, setup

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
                    "lazyarray",
                    "sultan",
                    "tqdm",
                    "scipy",
                    "gdown"
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
        "develop": ["pytest",
                "pytest-cov",
                "coverage",
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
