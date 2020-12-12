import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DreamDiff", 
    version="1.0.0",
    author="Blake Bullwinkel, Aditya Kumar, Carlos Robles",
    author_email="jbullwinkel@fas.harvard.edu, adityakumar@g.harvard.edu, carlosrobles@college.harvard.edu",
    description="A Python package for automatic differentiation, root-finding, optimization, and interpolation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/autodiffdreamteam/cs107-FinalProject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
