from distutils.core import setup

setup(
    name='advmdata',
    version='0.0.0dev1',
    packages=['advmdata'],
    url='https://github.com/mdomanski-usgs/linearmodel',
    license='CC0 1.0',
    author='Marian Domanski',
    author_email='mdomanski@usgs.gov',
    description='ADVM data handling',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python 3.6'
    ],
    python_requires='>=3', requires=['numpy', 'pandas', 'scipy']
)
