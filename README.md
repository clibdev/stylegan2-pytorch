# Fork of [rosinality/stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch)

Differences between original repository and fork:

* Compatibility with PyTorch >=2.6. (🔥)
* TensorFlow weights conversion to PyTorch using Docker. (🔥)
* Original pretrained models and converted ONNX models from GitHub [releases page](https://github.com/clibdev/stylegan2-pytorch/releases). (🔥)
* Model conversion to ONNX format using the [convert_onnx.py](convert_onnx.py) file. (🔥)
* Installation with [requirements.txt](requirements.txt) file.
* Added missing `dnnlib` directory from [NVlabs/stylegan2](https://github.com/NVlabs/stylegan2)
* Replaced CUDA kernels with native PyTorch operations.
* Sample script [generate_onnx.py](generate_onnx.py) for ONNX inference.
* The following deprecations and errors has been fixed:
  * UserWarning: range will be deprecated, please use value_range instead.
  * Exporting the operator 'aten::normal' to ONNX opset version 10 is not supported.

# Installation

```shell
pip install -r requirements.txt
```

# Pretrained models

* Download links:

| Method                    | Model Size (MB)         | weights                                                                                                                                                                                                                                                                                                                                            | SHA-256                                                                                                                                                                                                  |
|---------------------------|-------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| StyleGAN2-Car-Config-F    | 118.2<br>115.7<br>347.2 | [PyTorch](https://github.com/clibdev/stylegan2-pytorch/releases/latest/download/stylegan2-car-config-f.pt)<br>[ONNX](https://github.com/clibdev/stylegan2-pytorch/releases/latest/download/stylegan2-car-config-f.onnx)<br>[TensorFlow](https://github.com/clibdev/stylegan2-pytorch/releases/latest/download/stylegan2-car-config-f.pkl)          | 7dcf0b90bf687a75293ba8df8d923be34ee134b183029d9f426634b079f85c9d<br>2ff80461a6384eddcb318193a4d6d3a215fff446ae8144410b06f39315ed0db9<br>e1618eee2b3ce87c4a3849442f7850ef12a478556bff035c8e09ee7e23b3794c |
| StyleGAN2-Cat-Config-F    | 115.3<br>114.7<br>340.9 | [PyTorch](https://github.com/clibdev/stylegan2-pytorch/releases/latest/download/stylegan2-cat-config-f.pt)<br>[ONNX](https://github.com/clibdev/stylegan2-pytorch/releases/latest/download/stylegan2-cat-config-f.onnx)<br>[TensorFlow](https://github.com/clibdev/stylegan2-pytorch/releases/latest/download/stylegan2-cat-config-f.pkl)          | af199066967dcaa3bdf51e396084970c447799f593fcfb93d9c2c655475e39bb<br>84f215489a17c2eefa4986e7c4cf817db401aa546784515a36d716cd132a6dc1<br>08940fd616cfaf6bc7b0286b5d1a0b3f70febb26e136d64716c8d3f5e9bd3883 |
| StyleGAN2-Church-Config-F | 115.3<br>114.7<br>340.9 | [PyTorch](https://github.com/clibdev/stylegan2-pytorch/releases/latest/download/stylegan2-church-config-f.pt)<br>[ONNX](https://github.com/clibdev/stylegan2-pytorch/releases/latest/download/stylegan2-church-config-f.onnx)<br>[TensorFlow](https://github.com/clibdev/stylegan2-pytorch/releases/latest/download/stylegan2-church-config-f.pkl) | 59fd45db3f66219f947aa5273ea981fd543e899d00e6b8690b1be9c66d4a1a36<br>13cdfe32aeff340cf9afc1a6b8e1a3371d36f8e3708ff48c4b0b8dca2e87caf9<br>4090c5f83b44b83475c84864188aa39a0fb72694492302b7e6af36aad0d18104 |
| StyleGAN2-FFHQ-Config-F   | 126.6<br>116.0<br>364.0 | [PyTorch](https://github.com/clibdev/stylegan2-pytorch/releases/latest/download/stylegan2-ffhq-config-f.pt)<br>[ONNX](https://github.com/clibdev/stylegan2-pytorch/releases/latest/download/stylegan2-ffhq-config-f.onnx)<br>[TensorFlow](https://github.com/clibdev/stylegan2-pytorch/releases/latest/download/stylegan2-ffhq-config-f.pkl)       | 37dd0b9722af1788f016f1472416cfdbf6628777ddd8345d57cca15efdac4e89<br>3f6df78c88871723f0608996bd1621602c3a97145874479c533ac2cdbb059573<br>adf127ea7bb8a7788c8bdeda3c9937f7310b669b09ecf799ca53a631ff46948d |
| StyleGAN2-Horse-Config-F  | 115.3<br>114.7<br>340.9 | [PyTorch](https://github.com/clibdev/stylegan2-pytorch/releases/latest/download/stylegan2-horse-config-f.pt)<br>[ONNX](https://github.com/clibdev/stylegan2-pytorch/releases/latest/download/stylegan2-horse-config-f.onnx)<br>[TensorFlow](https://github.com/clibdev/stylegan2-pytorch/releases/latest/download/stylegan2-horse-config-f.pkl)    | c4bfc90b2e855dad434dafc02e74a4906b2f07fecd7006159147d45dc7c42fd4<br>608f3431c37c9a55fe9d583807b9549399908807c0f20889e4993c2fb954cf52<br>cbc73372c7d82d095913ca8b4b9f170560777175bd7c3b8a75dd55d70807f32b |

# Export from TensorFlow to PyTorch format

* Build image:

```shell
docker build -t stylegan2 .
```

* Convert weights:

```shell
docker run -it --rm -v ./:/opt/stylegan2 stylegan2 weights/stylegan2-car-config-f.pkl
docker run -it --rm -v ./:/opt/stylegan2 stylegan2 weights/stylegan2-cat-config-f.pkl
docker run -it --rm -v ./:/opt/stylegan2 stylegan2 weights/stylegan2-church-config-f.pkl
docker run -it --rm -v ./:/opt/stylegan2 stylegan2 weights/stylegan2-ffhq-config-f.pkl
docker run -it --rm -v ./:/opt/stylegan2 stylegan2 weights/stylegan2-horse-config-f.pkl

sudo chown -R $USER:$USER ./weights
```

# Inference

```shell
python generate.py --size 512 --ckpt weights/stylegan2-car-config-f.pt
python generate.py --size 256 --ckpt weights/stylegan2-cat-config-f.pt
python generate.py --size 256 --ckpt weights/stylegan2-church-config-f.pt
python generate.py --size 1024 --ckpt weights/stylegan2-ffhq-config-f.pt
python generate.py --size 256 --ckpt weights/stylegan2-horse-config-f.pt
```

* Generate curated FFHQ images (matches paper Figure 11):

```shell
python generate.py --size 1024 --seed 66 --ckpt weights/stylegan2-ffhq-config-f.pt
python generate.py --size 1024 --seed 230 --ckpt weights/stylegan2-ffhq-config-f.pt
python generate.py --size 1024 --seed 389 --ckpt weights/stylegan2-ffhq-config-f.pt
python generate.py --size 1024 --seed 1518 --ckpt weights/stylegan2-ffhq-config-f.pt
```

# Export to ONNX format

```shell
pip install onnx
```

```shell
python convert_onnx.py --size 512 --weights weights/stylegan2-car-config-f.pt
python convert_onnx.py --size 256 --weights weights/stylegan2-cat-config-f.pt
python convert_onnx.py --size 256 --weights weights/stylegan2-church-config-f.pt
python convert_onnx.py --size 1024 --weights weights/stylegan2-ffhq-config-f.pt
python convert_onnx.py --size 256 --weights weights/stylegan2-horse-config-f.pt
```

# ONNX inference

```shell
pip install onnxruntime
```

```shell
python generate_onnx.py --ckpt weights/stylegan2-car-config-f.onnx
python generate_onnx.py --ckpt weights/stylegan2-cat-config-f.onnx
python generate_onnx.py --ckpt weights/stylegan2-church-config-f.onnx
python generate_onnx.py --ckpt weights/stylegan2-ffhq-config-f.onnx
python generate_onnx.py --ckpt weights/stylegan2-horse-config-f.onnx
```

* Generate curated FFHQ images (matches paper Figure 11):

```shell
python generate_onnx.py --seed 66 --ckpt weights/stylegan2-ffhq-config-f.onnx
python generate_onnx.py --seed 230 --ckpt weights/stylegan2-ffhq-config-f.onnx
python generate_onnx.py --seed 389 --ckpt weights/stylegan2-ffhq-config-f.onnx
python generate_onnx.py --seed 1518 --ckpt weights/stylegan2-ffhq-config-f.onnx
```

# Reference

* [TalkUHulk/realworld-stylegan2-encoder](https://github.com/TalkUHulk/realworld-stylegan2-encoder)
* [NVlabs/stylegan2](https://github.com/NVlabs/stylegan2)
