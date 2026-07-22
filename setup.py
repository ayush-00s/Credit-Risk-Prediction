## this file Allows users to install package locally anywhere and run

from setuptools import find_packages,setup
from typing import List

## function for returning the list of requirement
HYPEN_E_DOT = '-e .' 
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)


setup(
    name = "CreditRiskPrediction",
    version='0.0.1',
    author="Ayush Tiwari",
    author_email="73ayushtiwari@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)