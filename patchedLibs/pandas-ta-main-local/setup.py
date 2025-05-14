from setuptools import setup
from setuptools import find_packages

long_description = "An easy to use Python 3 Pandas Extension with 130+ Technical Analysis Indicators. Can be called from a Pandas DataFrame or standalone like TA-Lib. Correlation tested with TA-Lib."

setup(
    name="pandas_ta",
    version="0.3.15",
    description=long_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kevin Johnson",
    author_email="appliedmathkj@gmail.com",
    url="https://github.com/twopirllc/pandas-ta",
    maintainer="Kevin Johnson",
    maintainer_email="appliedmathkj@gmail.com",
    download_url="https://github.com/twopirllc/pandas-ta.git",
    keywords=["technical analysis", "trading", "python3", "pandas"],
    license="The MIT License (MIT)",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Office/Business :: Financial",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    package_data={
        "data": ["data/*.csv"],
    },
    packages=find_packages(include=["pandas_ta*"]),     # automatically finds all your modules
    include_package_data=True,                          # picks up data files via MANIFEST.in
    install_requires=[
        "numpy>=2.0.0",
        "pandas>=2.1.4",
        "six==1.16.0"
    ],
    extras_require={
        "dev": [
            "alphaVantage-api", "matplotlib", "mplfinance", "scipy",
            "sklearn", "statsmodels", "stochastic",
            "talib", "tqdm", "vectorbt", "yfinance",
        ],
        "test": ["ta-lib"],
    },
)