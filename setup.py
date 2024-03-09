from distutils.core import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

project_urls = {
  'Documentation': 'https://dkudumovic1.github.io/distfeatselect/'
}


setup(name='distfeatselect',
      packages=['distfeatselect'],
      version='0.1.2',
      description='A Pyhton package implementing distributed feature selection algorithm.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/dkudumovic1/distfeatselect',
      download_url='https://github.com/dkudumovic1/distfeatselect/archive/refs/tags/0.1.2.tar.gz',
      author='Dzeneta Kudumovic',
      author_email='dzeneta.kudum@gmail.com',
      keywords=['feature selection', 'distributed feature selection', 'randomised feature selection'],
      license='MIT',
      project_urls=project_urls,

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
