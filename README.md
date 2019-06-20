# mridata-recon
Joseph Y. Cheng <jycheng [at] stanford [dot] edu>

Basic reconstruction scripts for data uploaded to [mridata.org](http://mridata.org)

## Setup
Install the required python modules:
```bash
pip install -r requirements
```

## Datasets
* [**Stanford 2D FSE**](http://mridata.org/list?project=Stanford%202D%20FSE): `recon_fse_2d.py` (also supports [**NYU machine learning data**](http://mridata.org/list?project=NYU%20machine%20learning%20data))

## Example
To create `cfl` and `hdr` files that are support by [BART](https://mrirecon.github.io/bart/), run the following command for the downloaded dataset `data.h5`:

```bash
python recon_fse_2d.py --verbose data.h5
```

For more options, run the following command:

```bash
python recon_fse_2d.py --help
```

## References
1. http://mridata.org/
2. https://mrirecon.github.io/bart/
