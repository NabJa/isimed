import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="meddist",
    version="0.0.1",
    author="Nabil Jabareen",
    author_email="nabil.jabareen@charite.de",
    description="Using physical distance as self-supervised learning signal.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/lukassen/whole-body-imaging/meddistssl",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
)
