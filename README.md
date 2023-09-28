# GSGEN: Text-to-3D using Gaussian Splatting

This repository contains the official implementation of [GSGEN: Text-to-3D using Gaussian Splattng](https://gsgen3d.github.io). 

### Video results


https://github.com/gsgen3d/gsgen/assets/44675551/6f3c0df7-e3d1-4a37-a617-8a35ade2d72d


https://github.com/gsgen3d/gsgen/assets/44675551/64e5662f-e8d5-4380-a2ac-540d6789f65b



https://github.com/gsgen3d/gsgen/assets/44675551/c9c25857-3b5c-4338-adb0-e8e0910c8260



https://github.com/gsgen3d/gsgen/assets/44675551/25e3a94e-5a3b-4e14-bdd5-bfbf44fc2b82




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
