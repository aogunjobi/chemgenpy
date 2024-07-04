---

# ChemGenPy

ChemGenPy is an advanced tool designed for the rapid and parallel generation of extensive molecular libraries, tailored for chemical and materials science applications. It enables users to efficiently create and manage large sets of molecules with customizable constraints and configurations.

## Features

- Generate large molecular libraries using building blocks and specified rules.
- Support for SMILES and SMARTS notation.
- Configurable constraints for generating molecules, including bonds, atoms, molecular weight, rings, and more.
- Synthetic feasibility analysis for generated molecules.
- Easy-to-use graphical user interface (GUI) built with ipywidgets.

## Installation

### Prerequisites

- Python 3.10 or higher
- Conda (recommended for managing dependencies)

### Clone the Repository

```bash
git clone https://github.com/aogunjobi/ChemGenPy.git
cd ChemGenPy
```

### Create a Conda Environment

```bash
conda create -n chemgenpy python=3.10
conda activate chemgenpy
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Running the GUI

To start the ChemGenPy GUI, navigate to the `notebooks` directory and run the `main.py` script:

```bash
cd chemgenpy/main
python main.py
```

### Command-Line Interface

ChemGenPy can also be run from the command line. Use the following syntax:

```bash
python chemgenpy.py -i <input_config_file> -b <building_blocks_file> -o <output_directory>
```

- `-i, --input`: Path to the configuration file.
- `-b, --building_blocks`: Path to the building blocks file.
- `-o, --output`: Directory to save the output files.


