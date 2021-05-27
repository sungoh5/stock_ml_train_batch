from setuptools import find_packages, setup

with open('.version') as version_file:
    version = version_file.read()

setup(
    name='stock-ml-training-batch',
    version=version,
    description='This service reads stock data and performs ML training using the RNN algorithm.',
    author='Sungoh Kim',
    author_email='sungoh5.kim@g.skku.edu',
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=[          
        'pymongo==3.11.4',
        'pandas==1.2.4',
        'scikit-learn==0.24.2',
        'tensorflow==2.2.0',
        'numpy==1.18.5'
        ],
    include_package_data=True
)
