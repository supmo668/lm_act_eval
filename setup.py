#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    # Core dependencies that are always installed
    'numpy==1.25.2',
    'langchain==0.0.224',
    'pandas==2.0.3',
    'pydantic==1.10.14',
    'openai==1.14.0',
    'Pillow==10.0.1',
    'multion==0.3.13',
    'playwright==1.37.0',
    'torch==2.0.1',
    'pydantic==1.10.14'
]

multion_requirements = [
    # Dependencies for multion component
    'multion==0.3.13',
    'deepeval==0.20.90',
    'lm_eval==0.4.2',
    'braintrust_core==0.0.26',
    'autoevals==0.0.53'
]

test_requirements = ['pytest>=3', ]

setup(
    author="Matthew M",
    author_email='matthew@multion.ai',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Evaluation for LLM Actions & Trajectories",
    entry_points={
        'console_scripts': [
            'lm_act_eval=lm_act_eval.cli:main',
        ],
    },
    install_requires=requirements,
    extras_require={
        'multion-eval': [
            # Dependencies for the extra1 component
            'multion==0.3.13',
            'deepeval==0.20.90',
            'lm_eval==0.4.2',
            'braintrust_core==0.0.26',
            'autoevals==0.0.53'
        ]
    },
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='lm_act_eval',
    name='lm_act_eval',
    packages=find_packages(include=['lm_act_eval', 'lm_act_eval.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/supermomo668/lm_act_eval',
    version='0.1.0',
    zip_safe=False,
)
