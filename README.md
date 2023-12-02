# `novelty-detection`

A Python package for novelty/anomaly detection methods using Self-Organizing Maps (SOM). The project follows a tree structure similar to this cookiecutter [template][cookiecutter-template]. The code follows is implemented following [this practices][this-practices] in order to have a common basic API for all the objects

## Getting started

Run the following command for clonning the repository from GitHub:

```shell
git clone https://github.com/opticsensors/novelty-detection.git
```

Follow the prompts; if you are asked to re-download `govcookiecutter`, input `yes`.
Default responses are shown in the squared brackets; to use them, leave your response
blank, and press enter.

Once you've answered all the prompts, your project will be created. Then:

1. Make sure  you have the latest version of pip and PyPA’s build installed:
   ```shell
   py -m pip install --upgrade pip
   py -m pip install --upgrade build
   ```

2. In your terminal, navigate to your new project, and initialise Git
   ```shell
   git init
   ```

3. Install the necessary packages using `pip`:
   ```shell
   py -m build
   py -m pip install -e . [dev]
   ```

4. Stage all your project files, and make your first commit
   ```shell
   git add .
   git commit -m "First commit"
   ```


## Licence

Unless stated otherwise, the codebase is released under the MIT License. This covers
both the codebase and any sample code in the documentation. The documentation is ©
Crown copyright and available under the terms of the Open Government 3.0 licence.


## Acknowledgements

[This template is based off the DrivenData Cookiecutter Data Science
project][drivendata]. Specifically, it uses a similar `src` folder structure,
and a modified version of the `help` commands in the `Makefile`s.

[cookiecutter-template]: https://github.com/drivendata/cookiecutter-data-science
[this-practices]: https://scikit-learn.org/stable/developers/develop.html