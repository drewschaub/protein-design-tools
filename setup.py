from setuptools import setup, find_packages

setup(
    name='protein-design-tools',
    version='0.1.0',
    author='Andrew Schaub',
    author_email='andrew.schaub@protonmail.com',
    description='A library of tools for protein design.',
    long_description=open('README.md').read(),
    url='https://github.com/drewschaub/protein-design-tools',
    license=' MIT License',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
    packages=find_packages(exclude=['tests*']),
    install_requires=['numpy'],  # List your project dependencies here
    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage'],
    },
)