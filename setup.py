from setuptools import setup, find_packages

setup(
    name="reefconnect",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'geopandas',
        'numpy',
        'xarray',
        # Add other dependencies
    ],
    entry_points={
        'console_scripts': [
            'get-angles=scripts.get_angles:main',
            'get-connectivity=scripts.get_connectivity:main',
            'get-kernels=scripts.get_kernels:main',
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for reef connectivity analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)