import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="thermal-base",
    version="1.0.0",
    description="Tools for operating on thermal images",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/detecttechnologies/thermal_base",
    author="Detect Technologies",
    author_email="support@detecttechnologies.com",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "Pillow",
        "logzero",
        "matplotlib",
        "numpy",
        "opencv_python",
        "tqdm",
    ],
)
