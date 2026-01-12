# GPROF IR

**GPROF IR** (`gprof_ir`) is a Python package for retrieving precipitation from geostationary **infrared window-channel observations**. It produces background precipitation fields for **merged precipitation products**, complementing passive-microwave retrievals from the GPROF-NN framework.

## Installation

Install `gprof_ir` directly from GitHub using `pip`:

```sh
pip install git+https://github.com/simonpf/gprof_ir
```

## Running Retrievals

The `gprof_ir retrieve` command operates on **GPM merged IR files**  
([GPM_MERGIR_1](https://disc.gsfc.nasa.gov/datasets/GPM_MERGIR_1/summary)).

### Running on a Single File

To process a single merged IR file, run:

```sh
gprof_ir retrieve merg_2020010100_4km-pixel.nc4 gprof_ir_2020010100.nc
```

This reads `merg_2020010100_4km-pixel.nc4` and writes the retrieval results to  
`gprof_ir_2020010100.nc` in NetCDF4 format.

### Controlling CPU Threads

By default, `gprof_ir` may use multiple CPU threads. To restrict the number of threads, use:

```sh
gprof_ir retrieve merg_2020010100_4km-pixel.nc4 --n_threads 1
```

### Retrieval Options

The behavior of `gprof_ir retrieve` can be customized using the following options:

| Option | Description |
|--------|-------------|
| `--device` | Compute device: `cpu`, `cuda:0`, …, `cuda:n` (for GPU acceleration). |
| `--dtype` | Floating-point precision used during inference. |
| `--variant` | Retrieval model variant: `cmb` (trained on GPM 2B-CMB) or `gmi` (trained on GPROF-V8 GMI retrievals). |
| `--output_format` | Output format: `netcdf` (default) or `binary`. |

## Running on Multiple Files

To process all merged IR files in a directory (including subdirectories), use:

```sh
gprof_ir run /path/to/input_directory --output_path /path/to/output_directory
```

### Batch Retrieval Options

The following options are available for `gprof_ir run`:

| Option | Description |
|--------|-------------|
| `--output_path` | Directory in which retrieval results are written. |
| `--device` | Compute device: `cpu`, `cuda:0`, …, `cuda:n`. |
| `--dtype` | Floating-point precision used during inference. |
| `--variant` | Retrieval model variant: `cmb` or `gmi`. |
| `--n_steps` | Number of input timesteps for the `gmi` variant (`1`, `3`, or `5`). |
| `--output_format` | Output format: `netcdf` (default) or `binary`. |

## Configuring the Model Path

GPROF IR automatically downloads its trained retrieval models from Hugging Face and caches them locally. By default, models are stored in the platform-specific user data directory (e.g., `~/.share/gprof_ir` on Linux).

To change the model directory, run:

```sh
gprof_ir config set_model_path /new/model/path
```
