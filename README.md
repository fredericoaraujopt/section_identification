# Section Identification

Describe the package
This Python package provides tools for targeting sections on silicon wafers.

## Installation
```
git clone https://github.com/fredericoaraujopt/section_identification.git
cd section_identification
pip install -e .
```
## Tutorial

Use the `demo.ipynb` notebook to explore the functionality of the package. This notebook provides step-by-step examples for identifying and targeting sections on silicon wafers.

To run the notebook:
1. Ensure you have Jupyter Notebook installed.
2. Open the notebook:
    ```
    jupyter notebook demo.ipynb
    ```
3. Follow the instructions within the notebook to execute the cells and interact with the examples.

# Basic git tutorial
## Contributing new changes
Make sure your main branch is up to date
```
git checkout main
git pull
```
Checkout new branch
```
git checkout -b <name>-<feature>
```
Make changes.
Make sure tests pass:
```
pytest .
```
Push changes:
```
git add <files to be pushed>
git commit -m "<commit message>"
git push # might need to use git push --set-upstream origin <name>-<feature>
```
## Pulling changes
Make sure your main branch is up to date
```
git checkout main
git pull
```
Checkout the branch that you want to update
```
git checkout <name>-<feature>
git merge main
```

