from setuptools import setup, find_packages

setup(
    name='shakhbanov_ml',
    version='0.1.0',
    author='Zurab Shakhbanov',
    author_email='zurab@shakhbanov.ru',
    description='ML tools for time series',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://shakhbanov.org',
    project_urls={
        'GitHub': 'https://github.com/shakhbanov/shakhbanov_ml',
    },
    download_url='https://github.com/shakhbanov/shakhbanov_ml/archive/refs/tags/v0.1.0.tar.gz',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy==1.24',
        'pandas',
        'matplotlib',
        'tqdm',
        'typing',
        'joblib',
        'prophet',
        'lightgbm',
        'holidays',
        'scikit-learn',
        'pykalman',
    ],
)
