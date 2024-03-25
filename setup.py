#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'beartype==0.17.2',
    'huggingface-hub==0.21.4',
    'langchain>=0.0.63',
    'multion==0.3.13',  # Note: Also listed in multion_requirements, ensure it's intentional
    'numpy==1.25.2',
    'openai==1.14.0',
    'pandas==2.2.1',
    'Pillow==10.0.1',
    'playwright==1.37.0',
    'pydantic==1.10.14',  # Duplicate removed
    'PyYAML==6.0.1',
    'python-dotenv==1.0.1',
    "scikit-image==0.22.0",
    'torch==2.2.1',
    "Requests==2.31.0"
    "wandb==0.16.4"
]

webarena_requirements = [
    "gymnasium==0.29.1",
    "matplotlib==3.8.3",
    "metrics==0.3.3",
    "trajectory==0.1.1",
    "vertexai==1.43.0",
    "text-generation==0.6.1",
    "aiolimiter==1.1.0"
]

multion_requirements = [
    'autoevals==0.0.53',
    'braintrust_core==0.0.26',
    'deepeval==0.20.90',
    'lm_eval==0.4.2',
    'multion==0.3.13',  # Also listed in core requirements, included here for component-specific context
] \
    + webarena_requirements




test_requirements = ['pytest>=3', ]

setup(
    name='lm_act_eval',
    version='0.1.0',
    packages=['lm_act_eval'],
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
            'lm_act_eval=lm_act_eval.__main__:main',
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
