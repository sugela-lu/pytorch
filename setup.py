from setuptools import setup, find_packages

setup(
    name="medical-super-resolution",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'numpy>=1.19.2',
        'Pillow>=8.0.0',
        'tqdm>=4.50.0',
        'matplotlib>=3.3.0',
        'flask>=2.0.0',
    ],
    python_requires='>=3.7',
)
