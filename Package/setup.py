from setuptools import setup, find_packages

setup(
    name='my_package',  # Replace with your package's name
    version='0.1',
    description='A package for light curve fitting using numerical optimization',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/my_package',  # Replace with your repository URL
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
