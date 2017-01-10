# If working in windows, uncomment the following two lines
# import tempfile
# tempfile.tempdir=os.getcwd()+"\\temp\\"

from setuptools import setup, find_packages
setup(
    name="inputbuilder",
    version="0.4",
    packages=['ratebuilder', 'spikebuilder'],
    author="Arjun Rao",
    author_email="arjun210493@gmail.com",
    description="This module provides the infrastructure to create custom (spike/rate) builders",
    license="MIT",
    keywords="Generic Builder builder generic",
    install_requires=['genericbuilder>=0.4', 'numpy', 'scipy'],
    provides=['ratebuilder', 'spikebuilder'],
    dependency_links=['git+ssh://git@git.tugraz.at/GenericBuilder.git@releases#egg=genericbuilder-0.4']
)
