# GSGEN: Text-to-3D using Gaussian Splatting

This repository contains the official implementation of [GSGEN: Text-to-3D using Gaussian Splattijg](https://gsgen3d.github.io). 


### Instructions:
1. Install the requirements:
```
pip install -r requirements.txt
```
2. Build the extension for Gaussian Splatting:
```
cd gs
./build.sh
```
3. Start training!
```python
python main.py --config-name=base prompt.prompt="<prompt>"
```
You can specify a different prompt for Point-E:
```python
python main.py --config-name=base prompt.prompt="<prompt>" init.prompt="<point-e prompt>"
```

### Viewer
Start the Viewer by:
```python
python vis.py <path-to-ckpt> --port <port>
```