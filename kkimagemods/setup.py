from setuptools import setup, find_packages

packages = find_packages(
        where='.',
        include=['kkimagemods*']
)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='kkimagemods',
    version='1.0',
    description='package template implemented in python',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kazukingh01/kkpackages/kkimagemods",
    author='Kazuki Kume',
    author_email='kazukingh01@gmail.com',
    license='Private License',
    packages=packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Private License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy>=1.18.2',
        'opencv-python>=4.2.0.34',
        'pandas>=1.0.3',
    ],
    python_requires='>=3.7'
)
