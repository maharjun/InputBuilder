# If working in windows, uncomment the following two lines
# import tempfile
# tempfile.tempdir=os.getcwd()+"\\temp\\"

from setuptools import setup
setup(
    name="inputbuilder",
    version="2.1.4",
    packages=['ratebuilder', 'spikebuilder'],
    author="Arjun Rao",
    author_email="arjun210493@gmail.com",
    description="This module provides the infrastructure to create custom (spike/rate) builders",
    license="MIT",
    keywords="Generic Builder builder generic",
    install_requires=['genericbuilder>=2.0.0', 'numpy', 'scipy'],
    provides=['ratebuilder', 'spikebuilder'],
    dependency_links=['git+https://github.com/maharjun/GenericBuilder.git@d9e9532#egg=genericbuilder-2.0.0']
)
