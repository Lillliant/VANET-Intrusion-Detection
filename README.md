# COMP-4990-Intrusion-Detection

This is a framework containing machine learning models for intrusion detection and misbehavior detection in vehicular networks.

## Overview

The framework currently contains the following models:

## Quick Start

### Installation

The following commands create a virtual environment in Python and installs the necessary dependencies using `pip`.

```bash
python -m venv vimi
source vimi/bin/activate
pip install -r requirements.txt
```

Alternatively, `conda` can be used to install the dependencies in a conda environment.

```bash
conda env create -f environment.yml
```

### Running the Pipeline

```bash
python src/main.py data/your_dataset.csv outputs
```

