import setuptools

setuptools.setup(
    name='UnsupSeg',               
    version='0.0.0',         
    packages=['UnsupSeg'],   
    include_package_data=True,  
    exclude_package_date={'':['.gitignore']},
    python_requires='>=3.8',
    install_requires=[
        'tqdm',
        'librosa',
        'dill>=0.3.3',
        'boltons==20.0.0',
        'hydra-core>=0.11.3',
        'numpy>=1.18.1',
        'pytorch-lightning>=0.7.6',
        'test-tube==0.7.5',
        'torch-optimizer==0.0.1a12',
        'torchaudio>=0.4.0',
        'tqdm',
        'wandb',
        'numba>=0.48',
    ]
)
