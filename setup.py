from setuptools import setup, find_packages

setup(
    name='fastsdcpu-pip',
    version='0.1.2',
    packages=find_packages(
        exclude=['test', 'docs']
    ),
    description='Fast stable diffusion on CPU and AI PC ported to a pip module',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Cuisset Mattéo',
    author_email='matteo.cuisset@gmail.com',
    url='https://github.com/Flyns157/fastsdcpu/pip',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    license="MIT AND (Apache-2.0 OR BSD-2-Clause)",
    license_files = ["THIRD-PARTY-LICENSES", "LICENSE"],
    python_requires='>=3.11',
    install_requires=open('requirements.txt').read().splitlines(),
)
