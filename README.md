# GPROF IR

The ``gprof_ir`` package provides a precipitation retrieval for geostationary
IR-window-channel observations to serve as background precipitation fields for
merged precipitation products.

## Installation

The ``gprof_ir`` package can be installed via pip:

``` shellsession
pip install git+https://github.com/simonpf/gprof_ir
```

## Running retrievals

The ``gprof_ir`` retrieval can be run directly on GPM [merged IR files](https://disc.gsfc.nasa.gov/datasets/GPM_MERGIR_1/summary). 

The retrieval can be run on a single file using

``` shellsession
gprof_ir retrieve merg_2020010100_4km-pixel.nc4
```

This will run the retrieval on the file ``merg_2020010100_4km-pixel.nc4`` and write the results to the current working directory.
Instead of specifying a specific input file, it is also possible to provide a folder to the ``gprof_ir retrieve`` command. The retrieval will then be run for all the merged-IR files found in the directory and its sub-directories.

> **Note**: ``OMP_NUM_THREADS=1 gprof_ir ...`` can be used to control the number of threads being used when running the retrieval on a CPU.

The following options are available to modify the default behavior of the ``gprof_ir retrieve`` command:
 
 - ``--output_path path``: Write retrieval results to ``path`` instead of the current working directory.
 - ``--device``: Run the retrieval on the following device (``cpu, cuda:0, ..., cuda:n``)
 - ``--start_time YYYY-mm-ddTHH:MM:SS``: If the retrieval is run on a directory of files, this optional will cause all files with a time stamp earlier than the given date to be ignored.
 - ``--end_time YYYY-mm-ddTHH:MM:SS``: If the retrieval is run on a directory of files, this optional will cause all files with a time stamp later than the given date to be ignored.


