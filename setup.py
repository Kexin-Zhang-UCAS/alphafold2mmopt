from setuptools import setup, find_packages

setup(
  name = 'alphafold2mmopt',
  packages=find_packages(),
  include_package_data=True,
  version = '0.2',
  description = 'optimize pdb structures using fine-tune training strategy in alphafold2',
  author = 'Kexin Zhang',
  author_email = 'zhangkx2@shanghaitech.edu.cn',
  license='MIT',
  keywords = ['computational biology', 'force fields', 'molecular mechanics',"protein"],
  classifiers = [
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        ],
  install_requires=['torch',"openmm","biopython","dm-haiku","dm-tree","pdbfixer"],
  entry_points={
        'console_scripts': [
            'alphafold2mmopt=alphafold2mmopt.run:main',
        ],
    },

)