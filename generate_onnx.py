import argparse
import numpy as np
import onnxruntime as ort
from PIL import Image


def normalization(data):
    _range = np.max(data) - np.min(data)

    return (data - np.min(data)) / _range


def generate(args, ort_session):
    rnd = np.random.RandomState(args.seed)
    sample_z = rnd.randn(1, 1, args.latent).astype(np.float32)

    outputs = ort_session.run([ort_session.get_outputs()[0].name],
                              {ort_session.get_inputs()[0].name: sample_z})[0]

    output = (outputs.squeeze().transpose((1, 2, 0)) + 1) / 2
    output[output < 0] = 0
    output[output > 1] = 1
    output = normalization(output) * 255
    output = Image.fromarray(output.astype('uint8'))

    output.save('sample/test.png')


if __name__ == "__main__":
    device = "cpu"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed for generating random numbers",
    )

    args = parser.parse_args()

    args.latent = 512

    ort_session = ort.InferenceSession(args.ckpt)

    generate(args, ort_session)
