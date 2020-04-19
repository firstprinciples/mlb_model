import setuptools
import fps_mlb_model

# Get the version of this package
version = fps_mlb_model.version

# Get the long description of this package
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='fps_mlb_model',
    version=version,
    author="FirstPrinciples Data Science",
    author_email="",
    description="MLB modeling project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://firstprinciples/mlb_model",
    packages=setuptools.find_packages(exclude=['unit_tests']),
    install_requires=['',],
    package_data={'fps_mlb_model': ['models/*.joblib']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
