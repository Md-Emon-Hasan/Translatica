from setuptools import setup
from setuptools import find_packages

setup(
    name='Translatica',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'transformers',
        'datasets',
        'evaluate',
        'sacrebleu',
        'torch',
        'pytest',
        'sentencepiece'.
        'gunicorn',
        'flask',
        'peft',
        'accelerate',
        'scipy',
        'numpy',
        'protobuf'
    ],
    description='A fine-tuned English to Spanish translation model using Flask and Hugging Face',
    author='Md Emon Hasan',
    author_email='iconicemon01@gmail.com',
    url='https://github.com/Md-Emon-Hasan/Translatica',  
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
