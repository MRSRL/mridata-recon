# mridata-recon
Joseph Y. Cheng <jycheng [at] stanford [dot] edu>

Basic reconstruction scripts for data uploaded to [mridata.org](http://mridata.org)

## Setup
Install the required python modules:
```bash
pip install -r requirements
```

## Datasets
* **Stanford 2D FSE**: `recon_fse_2d.py`

## Example
To create `cfl` and `hdr` files that are support by [BART](https://mrirecon.github.io/bart/), run the following command for the downloaded dataset `data.h5`:

```bash
python recon_fse_2d.py --verbose data.h5
```

## References
1. http://mridata.org/
2. https://mrirecon.github.io/bart/
