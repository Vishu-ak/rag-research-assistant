"""Setup configuration for the RAG Research Assistant package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="rag-research-assistant",
    version="1.0.0",
    author="RAG Research Team",
    author_email="team@example.com",
    description=(
        "A production-quality Retrieval-Augmented Generation (RAG) system "
        "for intelligent research assistance over document collections."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rag-research-assistant",
    packages=find_packages(exclude=["tests*", "notebooks*", "docker*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=24.0.0",
            "isort>=5.13.0",
            "mypy>=1.11.0",
            "flake8>=7.1.0",
            "pytest>=8.3.0",
            "pytest-asyncio>=0.24.0",
            "pytest-cov>=5.0.0",
        ],
        "gpu": [
            "torch>=2.3.0",
            "faiss-gpu>=1.7.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "rag-serve=api.main:run_server",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
