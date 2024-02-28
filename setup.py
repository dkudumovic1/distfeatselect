from distutils.core import setup

setup(name='distfeatselect',
      packages=['distfeatselect'],
      version='0.1.0',
      description='A Pyhton package implementing distributed feature selection algorithm.',
      url='https://github.com/dkudumovic1/distfeatselect',
      author='Dzeneta Kudumovic',
      author_email='dzeneta.kudum@gmail.com',
      keywords=['feature selection', 'distributed feature selection', 'randomised feature selection'],
      license='MIT',

      install_requires=[
          'numpy',
          'pandas',
          'statsmodels',
          'scikit-learn',
          'dcor',
      ],

      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
      ],
      )
