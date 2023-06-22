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
                        "scikit-learn",
                        "cardinality",
                        "lazyarray",
                        "sultan",
                        "tqdm",
                        "scipy",
                        "gdown"
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
)
