from setuptools import find_packages, setup


setup(
    name="hugme",
    version="0.0.1",
    author="HUN-REN Research Center for Linguistics",
    author_email="osvathm.matyas@nytud.hun-ren.hu",
    description="Library to train and evaluate models on HuGME benchmark.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="models model-training fine-tuning natural-language-processing deep-learning evaluation benchmark",
    package_dir={"": "src"},
    packages=find_packages("src"),
    entry_points={
        "console_scripts": ["hugme=cli:cli"],
    },
    python_requires=">=3.8.0",
    install_requires=open("requirements.txt", "r").read().splitlines(),
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    include_package_data=True,
)