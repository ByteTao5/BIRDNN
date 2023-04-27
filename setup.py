from setuptools import setup

setup(
    name='BIRDNN',
    version='',
    packages=[''],
    url='https://github.com/ByteTao5/BIRDNN',
    license='',
    author='',
    author_email='',
    description='',
    python_requires='>=3.7, <3.8',
    install_requires=[
	    'protobuf==3.18.1',
        'torch==1.10.0',
        'numpy',
        'scipy',
        'onnx==1.10.2',
        'onnxruntime==1.9.0',
        'onnx2pytorch==0.4.1',
        'diffabs==0.1',
        'matplotlib',
        'future',
        'tqdm',
        'h5py==3.8.0',
        'pyswarms==1.3.0'
    ],
)
