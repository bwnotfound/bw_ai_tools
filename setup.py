from setuptools import setup, find_packages


setup(
    name='bwai',
    author='bw',
    version='0.0.2',
    description='Tools created by bw',
    license='MIT License',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)
