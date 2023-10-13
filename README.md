# GSGEN: Text-to-3D using Gaussian Splatting

This repository contains the official implementation of [GSGEN: Text-to-3D using Gaussian Splattng](https://gsgen3d.github.io). 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kg8OOnVXSnnEIk9IYBg55ZqkPfMh14xf?usp=sharing)


### [Paper](https://arxiv.org/abs/2309.16585) | [Project Page](https://gsgen3d.github.io/)

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
#### splat viewer
We support [splat](https://github.com/antimatter15/splat) viewer now !
Click the captions of text-to-3D results in our project page to watch the assets in a WebGL based viwer.
[Example: a pineapple](https://gsgen3d.github.io/viewer.html?url=A_zoomed_out_DSLR_photo_of_DSLR_photo_of_a_pineapple.splat).
This great viewer achieves > 40 FPS on my MacBook with M1 pro chip.

#### viser based viewer (Visualize checkpoints on your own computer)
Start the Viewer by:
```python
python vis.py <path-to-ckpt> --port <port>
```
If you are training on servers, [tunneling the port using SSH](https://www.ssh.com/academy/ssh/tunneling-example)
```bash
ssh -L <your_local_port>:<your_server_ip>:<your_server_port> <your_username>@<your_server>
```
then open the viewer in your host computer on port `<your_local_port>`.

### Exports
First set the `PYTHONPATH` env var:
```bash
export PYTHONPATH="."
```
#### To `.ply` file
```bash
python utils/export.py <your_ckpt> --type ply
```
#### To `.splat` file
```bash
python utils/export.py <your_ckpt> --type splat
```
#### To mesh (Currenly only support shape export)
```bash
python utils/export.py <your_ckpt> --type mesh --batch_size 65536 --reso 256 --K 200 --thresh 0.1
```
where the <your_ckpt> can be the path to the .pt checkpoint file or, more conveniently, can be the id for the run (the display name of the run in wandb, e.g. `0|213630|2023-10-11|a_high_quality_photo_of_a_corgi`). The exported files are reside in the `exports/<export-type>`.

## Updates
- [2023-10-08] Now support exports to `.ply` and `.splat` files. Mesh exporting are coming soon.
- [2023-10-13] Now support Shap-E initialize, try it with `init.type="shap_e"`
  
### TODO
- [ ] Support full mesh export. (Coming soon)  
- [ ] Support [VSD loss](https://github.com/thu-ml/prolificdreamer). (The VSD code is already done, further tuning is on the way)
- [ ] Support more guidance, e.g. [zero123](https://zero123.cs.columbia.edu/), [make-it-3d](https://github.com/junshutang/Make-It-3D), [ControlNet Openpose](https://github.com/mhussar/Controlnet3DCharacterRotation/tree/main), etc.


## Acknowledgement
This code base is built upon the following awesome open-source projects:
- [Stable DreamFusion](https://github.com/ashawkey/stable-dreamfusion)
- [threestudio](https://github.com/threestudio-project/threestudio)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [splat](https://github.com/antimatter15/splat)
- [Point-E](https://github.com/openai/point-e/issues)
- [Shap-E](https://github.com/openai/shap-e)
- [Make-it-3D](https://github.com/junshutang/Make-It-3D)

Thanks the authors for their remarkable job !