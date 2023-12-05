from setuptools import setup, find_packages

setup(
    name="piplyr",
    version="1.4",
    packages=find_packages(),
    install_requires=[
        'pandas>=1.0.0',
        'numpy>=1.18.0',
        
    ],
    author="Seyed Pozveh",
    author_email="saskosask@gmail.com",
    description="dplyr - like data manipulation",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/economer/piplyr",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
