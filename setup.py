from setuptools import find_packages,setup
from typing import List

a = '-e .'

def get_requirement(file_path:str)->List[str]:
    ''''
    this function will return a list of requirements
    '''
    with open(file_path, 'r') as file:
        requirements = file.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if a in requirements:
            requirements.remove(a)
    return requirements



setup(
    name = 'ml-project',
    version='0.0.1',
    author='Dhyey',
    author_email='dhyeybhimani001@gmail.com',
    packages=find_packages(),
    install_requires=get_requirement('requirements.txt')
)