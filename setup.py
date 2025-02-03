from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT= '-e .'

def get_requirements(file_path: str)->List[str]:
    """
    This funtion will be the list of requirements packages.
    """

    with open(file_path, 'r') as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", '') for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)


setup(
    name="end_to_end_mlproject",
    version= "0.0.1",
    author= "Prashant",
    author_email= "pkg13224@gmail.com",
    packages= find_packages(),
    install_requires = get_requirements('requirements.txt')
)