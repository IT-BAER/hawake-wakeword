import platform
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# Build extras_requires based on platform
def build_additional_requires():
    # py_version = platform.python_version()[0:3].replace('.', "")
    # if platform.system() == "Linux" and platform.machine() == "x86_64":
    #     additional_requires=[
    #         f"speexdsp_ns @ https://github.com/dscripka/openWakeWord/releases/download/v0.1.1/speexdsp_ns-0.1.2-cp{py_version}-cp{py_version}-linux_x86_64.whl",
    #     ]
    # elif platform.system() == "Linux" and platform.machine() == "aarch64":
    #     additional_requires=[
    #         f"speexdsp_ns @ https://github.com/dscripka/openWakeWord/releases/download/v0.1.1/speexdsp_ns-0.1.2-cp{py_version}-cp{py_version}-linux_aarch64.whl",
    #     ],
    if platform.system() == "Windows" and platform.machine() == "x86_64":
        additional_requires = [
            'PyAudioWPatch'
        ]
    else:
        additional_requires = []

    return additional_requires


setuptools.setup(
    name="openwakeword",
    version="0.6.0",
    install_requires=[
        'onnxruntime>=1.10.0,<2',
        'ai-edge-litert>=2.0.2,<3; platform_system == "Linux" or platform_system == "Darwin"',
        'speexdsp-ns>=0.1.2,<1; platform_system == "Linux"',
        'tqdm>=4.0,<5.0',
        'scipy>=1.3,<2',
        'scikit-learn>=1,<2',
        'requests>=2.0,<3',
    ],
    extras_require={
        'test': [
                    'pytest>=7.2.0,<8',
                    'pytest-cov>=2.10.1,<3',
                    'pytest-flake8>=1.1.1,<2',
                    'flake8>=5.0,<7.1',
                    'pytest-mypy>=0.10.0,<1',
                    'types-requests',
                    'types-PyYAML',
                    'mock>=5.1,<6',
                    'types-mock>=5.1,<6',
                    'types-requests>=2.0,<3'
                ],
        'full': [
                    'mutagen>=1.46.0,<2',
                    'torch>=2.7.1,<3',
                    'torchaudio>=2.7.1,<3',
                    'torchinfo>=1.8.0,<2',
                    'torchmetrics>=1.2.0,<2',
                    'speechbrain>=1.0,<2',
                    'audiomentations>=0.30.0,<1',
                    'torch-audiomentations>=0.11.0,<1',
                    'tqdm>=4.64.0,<5',
                    'pytest>=7.2.0,<8',
                    'pytest-cov>=2.10.1,<3',
                    'pytest-flake8>=1.1.1,<2',
                    'pytest-mypy>=0.10.0,<1',
                    'acoustics>=0.2.6,<1',
                    'scipy>=1.7.0,<1.14.0',  # acoustics imports scipy.special.sph_harm (removed in 1.14)
                    'setuptools<81',  # a dependency of openwakeword.train imports pkg_resources (removed in 81)
                    'pyyaml>=6.0,<7',
                    'tensorflow==2.19.0',
                    'numpy>=1.26,<2',
                    'tensorflow_probability==0.25.0',
                    'protobuf>=4.25.1,<6',
                    'onnx2tf==1.28.8',
                    'tf-keras==2.19.0',
                    'onnx-graphsurgeon>=0.5.8,<1',
                    'sng4onnx>=1.0.2,<3',
                    'sne4onnx>=1.0.11,<3',
                    'psutil>=5.9.5,<8',
                    'ai-edge-litert>=2.1.2,<3',
                    'onnx>=1.22.0,<2',
                    'pronouncing>=0.2.0,<1',
                    'deep-phonemizer==0.0.19'
                ]
    },
    author="David Scripka",
    author_email="david.scripka@gmail.com",
    description="An open-source audio wake word (or phrase) detection framework with a focus on performance and simplicity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://pypi.org/project/openwakeword",
    project_urls={
        "Bug Tracker": "https://pypi.org/project/openwakeword/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0 License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
)