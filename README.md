# Section Identification

## Installation
```
git clone https://github.com/fredericoaraujopt/section_identification.git
cd section_identification
pip install -e .
```

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

