from setuptools import setup, find_packages

setup(
    name='bilingual-bridge',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'streamlit==1.20.0',
        'transformers==4.28.0',
        'datasets==2.12.0',
        'evaluate==0.3.0',
        'sacrebleu==2.3.1',
        'torch==2.1.0',
    ],
    description='A fine-tuned English to Spanish translation model using Streamlit and Hugging Face',
    author='Md Emon Hasan',
    author_email='iconicemon01@gmail.com',
    url='https://github.com/Md-Emon-Hasan',  
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
