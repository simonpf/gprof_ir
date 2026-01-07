# GPROF IR

**GPROF IR** (`gprof_ir`) is a Python package for precipitation retrievals from geostationary **infrared window-channel observations**. It provides background precipitation fields for **merged precipitation products** to complement passive microwave retrievals from the GPROF and GPROF-NN retrievals.

## Installation

`gprof_ir` can be installed directly from GitHub using `pip`:

```sh
pip install git+https://github.com/simonpf/gprof_ir
```

## Running Retrievals

The `gprof_ir retrieval` command can be run directly on  **GPM merged IR files** ([GPM_MERGIR_1](https://disc.gsfc.nasa.gov/datasets/GPM_MERGIR_1/summary)).

### Running on a Single File

To run the retrieval on a single **merged IR file**, use:

```sh
gprof_ir retrieve merg_2020010100_4km-pixel.nc4
```

This processes the file **`merg_2020010100_4km-pixel.nc4`** and writes the results to the **current working directory**.

### Running on a Directory

To process all **merged IR files** in a directory (including subdirectories):

```sh
gprof_ir retrieve /path/to/directory
```

### Controlling CPU Threads

By default, `gprof_ir` may use multiple threads when running on a **CPU**. You can limit the number of threads using:

```sh
OMP_NUM_THREADS=1 gprof_ir retrieve merg_2020010100_4km-pixel.nc4
```

### Retrieval Options

You can modify the default behavior of `gprof_ir retrieve` using the following options:

| Option | Description |
|--------|-------------|
| `--output_path path` | Saves retrieval results to **`path`** instead of the current directory. |
| `--device` | Specifies the device for computation: **`cpu`**, **`cuda:0`**, ..., **`cuda:n`** (for GPU acceleration). |
| `--dtype` | Specifies the floating point type to use for computation |
| `--variant` | Specifies the model variant to use for the retrieval: ``cmb`` for the retrieval trained on 2B-CMB precipitation or ``gmi`` for the model trained on GPROF V8 GMI retirevals.|
| `--n_steps` | Specifies the number of input timesteps to use for the ``gmi`` retrieval variant. Should be ``1``, ``3``, or ``5``.|
| `--output_format` | Specifies the output format to use for the results. ``netcdf`` for NetCDF4 format (the default) or ``binary`` for binary output format.|



## Configuring the model path

GPROF IR downloads the retrieval model from Hugging Face and stores it locally. By default, the model is saved in the appropriate user data directory (``~/.share/gprof_ir`` on linux).

To change the directory where ``gprof_ir`` stores and loads models, use the following command:

``` shellsession
gprof_ir config set_model_path /new/model/path
```
