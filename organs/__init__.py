"""Bundled PRISM organs — file-imported by ``prism_organ_loader``.

This file exists so setuptools recognises ``organs/`` as a package and
ships the directory inside the wheel. The loader resolves each organ via
``importlib.util.spec_from_file_location``, so no symbols need to be
re-exported here.
"""
