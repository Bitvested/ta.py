import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='ta_py',
     version='1.0.0',
     author="Nino Kroesen",
     author_email="ninokroesen@gmail.com",
     description="ta.py is a Python package for dealing with financial technical analysis",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/Bitvested/ta.py",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 2.7",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
