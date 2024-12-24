from setuptools import setup, find_packages

setup(
    name='protein-design-tools',
    version='0.1.28',
    description='A toolkit for protein structure analysis and design.',
    author='Andrew Schaub',
    author_email='andrew.schaub@protonmail.com',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/drewschaub/protein-design-tools',
    license=' MIT License',
    packages=find_packages(include=['protein_design_tools', 'protein_design_tools.*']),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
    package_dir={'protein_design_tools': 'protein_design_tools'},
    python_requires = ">=3.6",
    install_requires=['numpy', 'h5py'],  # List your project dependencies here
)
