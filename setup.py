from setuptools import setup, find_packages

setup(
    name='ragasEvaluator',
    version='0.1.0',
    author='Your Name',
    author_email='pjonny@gmail.com',
    description='Advanced evaluation framework for RAG systems with intelligent retry and dataset validation.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'PyYAML',
        'requests',
        # Add other dependencies as needed
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)