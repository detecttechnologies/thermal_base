# Base codes for Thermography

A python package for decoding and common processing for thermographs / thermograms. Currently supports FLIR's TIFF and PNG encoded metadata, and DJI-encoded metadata

[![Quality check](https://github.com/detecttechnologies/thermal_base/actions/workflows/qualitycheck.yml/badge.svg)](https://github.com/detecttechnologies/thermal_base/actions)

## Install
1. This tool requires exiftools to be installed.
    - Install with `sudo apt-get install exiftool`
2. Install this package with `pip install git+https://github.com/detecttechnologies/thermal_base.git@main`

## Usage
Import and use the package as follows:
```
from thermal_base.thermal_base import ThermalImage
from thermal_base import utils
...
...
image = ThermalImage(image_path="path/to/image", camera_manufacturer="dji/flir")
```

## Supported formats
|Data Format|Sample Cameras|Support|
|--|--|--|
|FLIR RJPG with TIFF-format of thermal embedding|Zenmuse XT-2|✅|
|FLIR RJPG with TIFF-format of thermal embedding|FLIR E-4, T660|✅|
|DJI-encoded thermal image|Zenmuse H20-T|✅|
|FLIR SEQ Thermal Video|Zenmuse XT-2|✅|
|FLIR CSQ Thermal Video|Zenmuse XT-2|❌|
>*RJPG is also known as R-JPEG

## Notes
* The use case for h20T camera can also be developed with dji-thermal-tool-analysis. Refer to [this link](https://exiftool.org/forum/index.php?topic=11401.0) to know more about the implementation. Also note that this method can be performed only with a 32 bit python interpreter and only on windows platform.

## Credits
* [Exiftool](https://exiftool.org/) is used to read metadata and raw values from thermal image files.
* The `flyr_unpack.py` file was derived from the [flyr library](https://bitbucket.org/nimmerwoner/flyr/src/master/).
* The conversion from raw to temperature values is done using raw2temp from [ThermImage-R](https://github.com/gtatters/Thermimage)
