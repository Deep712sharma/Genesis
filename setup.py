import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Genesis",
    version="v0.1.0",
    author="Deepanshi Sharma",
    description="Genesis: Evolving Optimal Cache Schedule for Diffusion Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Deep712sharma/Genesis.git",
    packages=setuptools.find_packages(exclude=["DeepCache.ddpm"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=['torch', 'diffusers', 'transformers'],
    python_requires='>=3.10',
)