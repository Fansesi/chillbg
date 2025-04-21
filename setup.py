from setuptools import setup, find_packages

setup(
    name='chillbg',  # Replace with your actual project name
    version='0.1.0',
    
    # Automatically discover and include all packages
    packages=find_packages(),
    
    # Optional but recommended metadata
    author='Burak Sina Akbudak',
    author_email='',
    description='Chill background generator.',
    long_description=open('README.md').read() if open('README.md').read() else '',
    long_description_content_type='text/markdown',
    
    # Dependencies (if any)
    install_requires=[
        # List your project dependencies here
        # e.g., 'numpy>=1.18.0',
        # e.g., 'pandas',
        "pillow>=10.4",
        "numpy>=1.26",
        "numba>=0.61.2"

    ],
    
    # Python version requirements
    python_requires='>=3.11',
    
    # Optional classification metadata
    classifiers=[
        'Programming Language :: Python :: 3.11',
    ]
)
