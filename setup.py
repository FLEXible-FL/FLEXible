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
    install_requires=["numpy", "pytest"],
    python_requires=">=3.9.0",
)
