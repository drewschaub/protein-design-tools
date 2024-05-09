from setuptools import setup, find_packages

setup(
    name='protein-design-tools',
    version='0.1.3',
    author='Andrew Schaub',
    author_email='andrew.schaub@protonmail.com',
    description='A library of tools for protein design.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
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
    package_dir = {"": "src"},
    packages=find_packages(where="src"),
    python_requires = ">=3.6",
    install_requires=['numpy'],  # List your project dependencies here
)