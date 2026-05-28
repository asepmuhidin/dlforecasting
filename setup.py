from setuptools import setup, find_packages

setup(
    name='dlforecasting',
    version='0.1.0',
    author='Asep Muhidin',
    description='Deep Learning Time Series Forecasting Package',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'pandas',
        'scikit-learn',
        'tqdm'
    ],
)