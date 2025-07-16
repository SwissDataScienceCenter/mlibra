# MALDI Experiment

This folder contains the MALDI experiment, which demonstrates the use of the l3di library along with experiment-specific classes and logic.

## Structure
- `experiment.py`: Main experiment logic and classes.
- `config.py`: Configuration for the experiment (parameters, paths, etc.).
- `utils.py`: Helper functions for the experiment.
- `example.py`: Example script showing how to run the experiment using l3di.
- `data/`: Folder for experiment-specific input data.
- `results/`: Folder for experiment outputs/results.

## Usage

1. Install dependencies (if any):
   ```bash
   pip install -r requirements.txt
   ```
2. Run the example:
   ```bash
   python example.py
   ```

## Requirements
- Python 3.8+
- l3di (imported from the parent package)

## Notes
- Place any input data in the `data/` folder.
- Outputs will be saved in the `results/` folder. 