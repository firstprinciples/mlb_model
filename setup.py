import setuptools
import apeel_mlb_model

# Get the version of this package
version = apeel_mlb_model.version

# Get the long description of this package
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='apeel_mlb_model',
    version=version,
    author="Apeel Data Science",
    author_email="software@apeelsciences.com",
    description="MLB modeling project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/apeelsciences/datascience/projects/mlb_model",
    packages=setuptools.find_packages(exclude=['unit_tests']),
    install_requires=['apeel_datatools',],
    package_data={'apeel_mlb_model': ['models/*.joblib']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
