import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='ta.py',
     version='1.0.0',
     scripts=['ta'] ,
     author="Nino Kroesen",
     author_email="ninokroesen@gmail.com",
     description="A Docker and AWS utility package",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/Bitvested/ta.py",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )

# create package tutorial = https://dzone.com/articles/executable-package-pip-install
