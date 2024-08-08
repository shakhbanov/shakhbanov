from setuptools import setup, find_packages

setup(
    name='shakhbanov',
    version='1.2.3',
    author='Zurab Shakhbanov',
    author_email='zurab@shakhbanov.ru',
    description='ML tools for time series',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://shakhbanov.org',
    project_urls={
        'GitHub': 'https://github.com/shakhbanov/shakhbanov',
    },
    download_url='https://github.com/shakhbanov/shakhbanov/archive/refs/heads/main.zip',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
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
