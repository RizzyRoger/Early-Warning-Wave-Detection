# FOWD
Processing framework for FOWD, a free ocean wave dataset.

## Installation

After downloading the repository, you can install FOWD and all dependencies.

```bash
$ pip install -r requirements.txt
$ pip install .
```

## Usage

After installing the Python code, you can use the command line tool `fowd` to create a FOWD dataset from a raw source.

### CDIP

Use [CDIP buoy data](https://cdip.ucsd.edu/):

```bash
$ fowd process-cdip 433p1 -o fowd-cdip-out
```

will process all CDIP data located in the `433p1` folder.

### Generic inputs

Use

```bash
$ fowd process-generic infile.nc -o outdir
```

Generic inputs must be netCDF files with the following structure:

```
Variables:
- time
- displacement

Attributes:
- sampling_rate
- water_depth
- longitude
- latitude
```

## QC plots

All data processing writes QC information in JSON format. You can visualize records in that QC file by using

```bash
$ fowd plot-qc qcfile.json
```

## Discovery reports

Train a rogue-wave model on a FOWD catalogue and list places with high predicted risk:

```bash
$ pip install ".[pipeline]"
$ fowd report generate --synthetic -o fowd-reports
$ python3 scripts/long_train.py --rounds 5000 --save-every 1000 -o fowd-reports
$ fowd report train --synthetic --rounds 5000 --save-every 1000 -o fowd-reports
$ fowd report list -o fowd-reports
$ fowd report filter --most-distinct -o fowd-reports --top 20
$ fowd report filter --actual -o fowd-reports --top 20
$ fowd report show RUN_ID -o fowd-reports
$ fowd report map -o fowd-reports --actual
# writes runs/<id>/world_map.png and opens a matplotlib window
$ fowd report map --at 2018-01-01T04:00 -o fowd-reports
$ fowd report map next -o fowd-reports
$ fowd report watch --delay 0.4 -o fowd-reports
```

`generate` also accepts `--input catalogue.nc`, `--cdip-folder`, or `--generic-infile`. `filter` ranks wave windows by predicted rogue-wave probability (places that look like they would have a rogue wave).

## Testing

Run tests and sanity checks via

```bash
$ fowd run-tests
```

Test results are checked automatically, but sanity checks have to be inspected manually.
