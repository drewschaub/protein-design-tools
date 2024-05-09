from setuptools import setup, find_packages

setup(
    name='protein-design-tools',
    version='0.1.5',
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
        'Programming Language :: Python :: 3',
    ],
    packages=['protein_design_tools'],
    package_dir={'protein_design_tools': 'protein_design_tools'},
    python_requires = ">=3.6",
    install_requires=['numpy'],  # List your project dependencies here
)