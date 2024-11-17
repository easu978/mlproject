from setuptools import find_packages,setup
from typing import List
def get_requirements(file_path:str)-> List[str]:
    '''
    This function will return a list of the requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()

setup(
name='mlproject',
version='0.0.1',
author='Asuk',
author_email='easuk007@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')

)




