# ramsey_netcom
Methods for the analysis of Ramsey Network Communities

Papers

Each manuscript has its own directory, holding its source, its measured data and
the script that turns one into the other.

`local_multi_path_2026/` -- "Local network evolution rules drive shortest path
multiplicity". `generate_data_and_figs.py` reproduces Figs. 1-4 and 6, and
`diamond_lattice_fig.py` reproduces Fig. 5.

`sqrt/` -- community scaling analysis; see the Makefile in that directory.

Reproducing the figures

Run from inside the paper directory, in the `gt` environment described below.
The script imports the package as `ramsey_netcom`, so the directory *containing*
the clone must be on the import path, and the clone itself must keep the name
`ramsey_netcom`:

cd local_multi_path_2026
PYTHONPATH=../.. python3 generate_data_and_figs.py

By default this replots Figs. 1-4 and 6 from the cached measurements in `data/`,
which takes seconds. The measurements themselves are produced by setting
`GENERATE_DATA = True` at the top of the script; that path re-runs the network
generation and the community inference from scratch and takes days, because each
point is an average over 100 realizations. The cached CSVs are committed so that
the figures can be checked without paying that cost.

The real-network panels of Fig. 6 read `data/as733_metrics.csv`,
`data/biogrid_interactome_metrics.csv` and `data/condmat_coauthor_metrics.csv`,
derived from the sources cited in the manuscript.

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
