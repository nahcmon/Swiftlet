"""Shrink Fluid Speech Pro's Core ML encoder from 8-bit weights to a 4-bit palette (lookup table).

usage: palettize_encoder.py <source Encoder.mlmodelc> <destination Encoder.mlmodelc>

The upstream encoder stores each weight matrix as int8 with one scale per output channel
(constexpr_blockwise_shift_scale). This rewrites every matrix with at least 65,536 weights as
constexpr_lut_to_dense: 16 fp16 values per output row, found by 1-D k-means over that row's dequantized
weights, plus a 4-bit index per weight. Smaller tensors and every other blob are copied unchanged.

Why a palette and not int4: the Neural Engine keeps palettized weights compressed in memory (about 380 MB
for this model, against 600 MB for int8), while block-wise int4 is expanded to fp16 at load. 3-bit palettes
use the same Neural Engine memory as 4-bit and lose accuracy, so 4-bit is the floor worth shipping.

The result is deterministic for the same input and NumPy build. Run with coremltools 9.0 and NumPy.
"""
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from coremltools import libmilstoragepython as ms

BLOB = r'BLOBFILE\(path = string\("@model_path/weights/weight\.bin"\), offset = uint64\((\d+)\)\)'
QUANT = re.compile(
    r'tensor<fp16, \[([0-9, ]*)\]> (\w+) = constexpr_blockwise_shift_scale\(data = tensor<int8, \[([0-9, ]*)\]>\('
    + BLOB + r'\), scale = tensor<fp16, \[([0-9, ]*)\]>\(' + BLOB + r'\)\)')
GENERIC = re.compile(r'tensor<(\w+), \[([0-9, ]*)\]>\(' + BLOB + r'\)')
MIN_ELEMENTS = 65536
BITS = 4
MARKER = 'NEWBLOB\x00'  # rewritten blobs carry this until the copy pass has run


def shape(text):
    return [int(x) for x in text.split(',') if x.strip()]


def blob(offset):
    return f'{MARKER}(path = string("@model_path/weights/weight.bin"), offset = uint64({offset}))'


def kmeans_rows(w, k, iters=20):
    """1-D k-means per row of w [rows, n], starting from quantiles; returns (fp16 centers [rows, k], idx [rows, n])."""
    rows, n = w.shape
    centers = np.quantile(w, (np.arange(k) + 0.5) / k, axis=1).T.astype(np.float64)
    # Offset each row so one searchsorted over the flattened array assigns every row at once.
    offsets = (np.arange(rows) * (float(np.abs(w).max()) * 4 + 1))[:, None]
    flat = (w + offsets).ravel()
    row_base = np.arange(rows)[:, None]

    def assign(c):
        bounds = (c[:, 1:] + c[:, :-1]) / 2 + offsets
        idx = np.searchsorted(bounds.ravel(), flat).reshape(rows, n) - row_base * (k - 1)
        return np.clip(idx, 0, k - 1)

    for _ in range(iters):
        centers.sort(axis=1)
        idx = assign(centers)
        lin = (idx + row_base * k).ravel()
        sums = np.bincount(lin, weights=w.ravel(), minlength=rows * k).reshape(rows, k)
        counts = np.bincount(lin, minlength=rows * k).reshape(rows, k)
        centers = np.where(counts == 0, centers, sums / np.maximum(counts, 1))
    centers.sort(axis=1)
    centers16 = centers.astype(np.float16)
    return centers16, assign(centers16.astype(np.float64)).astype(np.uint8)


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns('weight.bin'))
    reader = ms._BlobStorageReader(str(src / 'weights/weight.bin'))
    writer = ms._BlobStorageWriter(str(dst / 'weights/weight.bin'))
    text = (src / 'model.mil').read_text()
    counts = {'palette': 0, 'int8': 0}

    def write_fp16(arr):
        return writer.write_fp16_data(np.ascontiguousarray(arr.astype(np.float16).ravel()).view(np.uint16))

    def quant(m):
        out_shape, name, data_shape, d_off, s_shape, s_off = m.groups()
        dims, sdims = shape(data_shape), shape(s_shape)
        q = reader.read_int8_data(int(d_off)).astype(np.float32).reshape(dims)
        s = reader.read_fp16_data(int(s_off)).view(np.float16).astype(np.float32).reshape(sdims)
        if q.size < MIN_ELEMENTS:
            counts['int8'] += q.size
            return (f'tensor<fp16, [{out_shape}]> {name} = constexpr_blockwise_shift_scale(data = tensor<int8, [{data_shape}]>('
                    f'{blob(writer.write_int8_data(np.ascontiguousarray(q.astype(np.int8).ravel())))}), scale = tensor<fp16, [{s_shape}]>('
                    f'{blob(write_fp16(s))}))')
        rows = dims[0]
        centers, idx = kmeans_rows((q * s).reshape(rows, -1), 2 ** BITS)
        lut_dims = ', '.join(str(d) for d in [rows] + [1] * (len(dims) - 1) + [2 ** BITS, 1])
        counts['palette'] += q.size
        return (f'tensor<fp16, [{out_shape}]> {name} = constexpr_lut_to_dense(indices = tensor<uint{BITS}, [{data_shape}]>('
                f'{blob(writer.write_uint4_data(np.ascontiguousarray(idx.reshape(dims).ravel())))}), lut = tensor<fp16, [{lut_dims}]>('
                f'{blob(write_fp16(centers))}))')

    text = QUANT.sub(quant, text)

    readers = {'fp16': reader.read_fp16_data, 'fp32': reader.read_float_data, 'int8': reader.read_int8_data,
               'uint8': reader.read_uint8_data, 'int32': reader.read_int32_data, 'int16': reader.read_int16_data,
               'uint16': reader.read_uint16_data}
    writers = {'fp16': writer.write_fp16_data, 'fp32': writer.write_float_data, 'int8': writer.write_int8_data,
               'uint8': writer.write_uint8_data, 'int32': writer.write_int32_data, 'int16': writer.write_int16_data,
               'uint16': writer.write_uint16_data}

    def copy(m):
        dtype, dims, off = m.groups()
        data = readers[dtype](int(off))
        return f'tensor<{dtype}, [{dims}]>({blob(writers[dtype](np.ascontiguousarray(data)))})'

    text = GENERIC.sub(copy, text).replace(MARKER, 'BLOBFILE')
    del writer  # flushes weight.bin
    (dst / 'model.mil').write_text(text)
    size = (dst / 'weights/weight.bin').stat().st_size
    print(f'{dst}: {size / 2**20:.1f} MB, {counts["palette"] / 1e6:.1f}M weights palettized, '
          f'{counts["int8"] / 1e6:.2f}M kept 8-bit')


if __name__ == '__main__':
    main()
