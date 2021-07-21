# Base codes for Thermography

A python package for decoding and common processing for thermographs / thermograms. Currently supports FLIR's TIFF and PNG encoded metadata, and DJI-encoded metadata

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

**Disclaimer**

Tested against data from: 
* Flir encoding format: DJI XT-2, Flir E-40 (Temperature values go out of the image frame if the image height is <255 pixels)
* DJI-encoding format: DJI Zenmuse H20-T
* Thermapp (available in past commits)

## Notes
* The use case for h20T camera can also be developed with dji-thermal-tool-analysis.Refer to the link [https://exiftool.org/forum/index.php?topic=11401.0] to know more about the implementation. Also note that this method can be performed only with a 32 bit python interpreter and only on windows platform.
* The `flyr_unpack.py` file was derived from the [flyr library](https://bitbucket.org/nimmerwoner/flyr/src/master/).