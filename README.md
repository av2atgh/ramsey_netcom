# ramsey_netcom
Methods for the analysis of Ramsey Network Communities

Requirements

This module requires graph-tool. The easiest option is to use conda

conda create --name gt -c conda-forge graph-tool
conda activate gt

For other installation options follow instructions at https://graph-tool.skewed.de/static/doc/index.html

Cython-accelerated generators (optional)

The generators `generator_local_search`, `generator_bubbles`, and
`generator_dup_split` have Cython ports in `generators_fast.pyx` (10-25x faster).
Build the extension once, inside the same conda env you run from:

conda activate gt
pip install cython
python3 setup_fast.py build_ext --inplace

Once built, `libs.py` uses the compiled versions automatically; if the extension
is missing it falls back to the pure-Python implementations. The compiled module
is Python-version-specific, so rebuild it whenever you change or recreate the env.

NOTE: the Cython versions use a C++ RNG, so a given `seed` produces different
graphs than the pure-Python versions. Regenerate any seeded/cached datasets after
switching rather than mixing old and new output.
