import os
import setuptools

root_dir = os.path.dirname(os.path.realpath(__file__))

# dependencies
INSTALL_REQUIRES = [
    "torch",
    "matplotlib",
    "tqdm",
]

setuptools.setup(
    name="torchgp",
    version="0.0.1",
    author="TorchGP Developers",
    author_email="wchen.robotics@outlook.com",
    description="Simple PyTorch implementation of Gaussian process models for research and education",
    long_description=open(os.path.join(root_dir, "README.md")).read(),
    long_description_content_type="text/markdown",
    keywords=[
        "machine", "learning", "gaussian", "process", "pytorch"
    ],
    python_requires='>=3.8',
    install_requires=INSTALL_REQUIRES,
    url="https://github.com/Weizhe-Chen/TorchGP",
    packages=setuptools.find_packages(exclude=['tests']),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        'Programming Language :: Python :: 3.8',
    ],
    license='MIT',
    project_urls={
        "Documentation": "https://torchgp.readthedocs.io",
        "Repository": "https://github.com/Weizhe-Chen/TorchGP",
        "Bug Tracker": "https://github.com/Weizhe-Chen/TorchGP/issues",
        "Discussions": "https://github.com/Weizhe-Chen/TorchGP/discussions",
    },
)
