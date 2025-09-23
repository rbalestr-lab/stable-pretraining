Contributing
============

Thank you for your interest in contributing to ``stable-pretraining``!
This library is a community-driven project, and we greatly appreciate contributions of all kinds.

If you encounter any issues or have suggestions, please open an issue on our `issue tracker <https://github.com/rbalestr-lab/stable-pretraining/issues>`_. This allows us to address problems and gather feedback from the community.

For those who want to contribute code or documentation, you can submit a pull request. Below, you will find details on how to prepare and submit your pull request effectively.


PR Tutorial
-----------

The preferred workflow for contributing to ``stable-pretraining`` is to fork the
`main repository <https://github.com/rbalestr-lab/stable-pretraining>`_ on
GitHub, clone, and develop on a branch. Steps:

1. Fork the `project repository <https://github.com/rbalestr-lab/stable-pretraining>`_
   by clicking on the 'Fork' button near the top right of the page. This creates
   a copy of the code under your GitHub user account. For more details on
   how to fork a repository see `this guide <https://help.github.com/articles/fork-a-repo/>`_.

2. Clone your fork of the ``stable-pretraining`` repo from your GitHub account to your local disk::

      $ git clone git@github.com:YourLogin/stable-pretraining.git
      $ cd stable-pretraining

3. Install the package in editable mode with the development dependencies, as well as the pre-commit hooks that will run on every commit::

      $ pip install -e .[dev,doc] && pre-commit install

4. Create a ``feature`` branch to hold your development changes::

      $ git checkout -b my-feature

   Always use a ``feature`` branch. It's good practice to never work on the ``master`` branch!

5. Develop the feature on your feature branch. Add changed files using ``git add`` and then commit the changes using ``git commit``::

      $ git add modified_files
      $ git commit -m "Your commit message here"

   To record your changes in Git, then push the changes to your GitHub account with::

      $ git push -u origin my-feature

6. Follow `these instructions <https://help.github.com/articles/creating-a-pull-request-from-a-fork>`_
   to create a pull request from your fork. Then, a project maintainer will review your changes.


PR Checklist
------------

When preparing the PR, please make sure to
check the following points:

- The automatic tests pass on your local machine. This can be done by running ``python -m pytest stable_pretraining/tests -m unit`` in the root directory of the repository after making the desired changes. Only unit tests should be run locally. See `TESTING.md <https://github.com/rbalestr-lab/stable-pretraining/blob/main/TESTING.md>`_ for detailed information about writing and running tests.
- If your pull request addresses an issue, please use the pull request title to describe the issue and mention the issue number in the pull request description. This will make sure a link back to the original issue is created.
- The documentation is updated if necessary. You can edit the documentation using any text editor and then generate the HTML output by typing ``make html`` from the ``docs/`` directory. The resulting HTML files will be placed in ``docs/build/html/`` and are viewable in a web browser.

When creating a pull request (PR), use the appropriate prefix to indicate its status and type:

**Status prefix:**

- ``[WIP]`` **(Work in Progress)**: Use this prefix for an incomplete contribution where further work is planned before seeking a full review. Consider including a `task list <https://github.com/blog/1375-task-lists-in-gfm-issues-pulls-comments>`_ in your PR description to outline planned work or track progress. Remove the ``[WIP]`` prefix once the PR is ready for review.

**Type prefix (use when PR is ready):**

- ``[BugFix]``: For bug fixes and patches
- ``[Feature]``: For new features or enhancements
- ``[Example]``: For adding or updating examples
- ``[Benchmark]``: For benchmark-related changes
- ``[Doc]``: For documentation updates

A ``[WIP]`` PR can serve multiple purposes:
1- Indicate that you are actively working on something to prevent duplicated efforts by others.
2- Request early feedback on functionality, design, or API.
3- Seek collaborators to assist with development.


Testing
-------

All contributions should include appropriate tests. See `TESTING.md <https://github.com/rbalestr-lab/stable-pretraining/blob/main/TESTING.md>`_ for comprehensive guidance on writing unit tests, integration tests, and best practices for the stable-pretraining testing framework.


New contributor tips
--------------------

A great way to start contributing to ``stable-pretraining`` is to pick an item
from the list of `good first issues <https://github.com/rbalestr-lab/stable-pretraining/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22>`_ in the issue tracker. Resolving these issues allows you to start
contributing to the project without much prior knowledge.
