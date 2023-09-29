# GSGEN: Text-to-3D using Gaussian Splatting

This repository contains the official implementation of [GSGEN: Text-to-3D using Gaussian Splattijg](https://gsgen3d.github.io). 

[Paper](https://arxiv.org/abs/2309.16585)


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

## Acknowledgement
This code base is built upon the following awesome open-source projects:
[Stable DreamFusion](https://github.com/ashawkey/stable-dreamfusion)
[threestudio](https://github.com/threestudio-project/threestudio)
[3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
Thanks the authors for their remarkable job !