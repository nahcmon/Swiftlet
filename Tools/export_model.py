"""Export Fluid Speech Pro's Core ML weights into the encoder.bin and decoder.bin files Swiftlet reads.

usage: export_model.py <palettized Encoder.mlmodelc> <dir with Decoder.mlmodelc and JointDecision.mlmodelc> <out dir>

Writes <out dir>/encoder.bin (SpeechEncoder, Metal) and <out dir>/decoder.bin (SpeechDecoder, CPU).
Both are one little-endian file: a u64 index length, the JSON index, padding to a 16 KiB boundary (the data start),
then each tensor at a 256-byte aligned offset from the data start.

encoder.bin keeps the big matrices 4-bit: per row 16 fp16 palette values, and the indices in the lane-tiled layout
the GEMM kernels load directly (see wq2_layout). Everything else is fp16. The graph is walked in op order, so names
come from position, not from MIL names.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from coremltools import libmilstoragepython as ms


def write(path, fmt, tensors):
    index, blobs, offset = {}, [], 0
    for name, (kind, value) in tensors.items():
        if kind == 'f16':
            entry = {'type': 'f16', 'shape': list(value.shape), 'offset': offset}
            parts = [np.ascontiguousarray(value, dtype=np.float16).tobytes()]
        else:
            idx, table = value
            packed = wq2_layout(np.ascontiguousarray(idx, dtype=np.uint8))
            entry = {'type': 'lut4', 'shape': list(idx.shape), 'offset': offset,
                     'lut': offset + (len(packed) + 255) // 256 * 256}
            parts = [packed, b'\0' * ((-len(packed)) % 256), np.ascontiguousarray(table, dtype=np.float16).tobytes()]
        size = sum(len(x) for x in parts)
        blobs += parts + [b'\0' * ((-size) % 256)]
        index[name] = entry
        offset += size + (-size) % 256
    header = json.dumps({'format': fmt, 'tensors': index}, separators=(',', ':')).encode()
    start = (8 + len(header) + 16383) // 16384 * 16384
    with open(path, 'wb') as f:
        f.write(len(header).to_bytes(8, 'little') + header + b' ' * (start - 8 - len(header)))
        for b in blobs:
            f.write(b)
    print(f'{path}: {offset / 2**20:.1f} MB, {len(index)} tensors')


def wq2_layout(idx):
    """4-bit indices [N][K] -> u32 words [N/8][K/32][32 lanes]: each SIMD lane gets the nibbles of its 8x8 MMA
    fragments, byte s = rows (n, n+1) at column k = kb*32 + s*8 + m (fragment row m, column pair n)."""
    N, K = idx.shape
    assert N % 8 == 0 and K % 32 == 0, (N, K)
    lane = np.arange(32); q = lane // 4
    fm = (q & 4) + ((lane // 2) % 4); fn = (q & 2) * 2 + (lane % 2) * 2
    I = idx.reshape(N // 8, 8, K // 32, 4, 8)                       # nb, nn, kb, s, mm
    lo = I[:, fn, :, :, fm]; hi = I[:, fn + 1, :, :, fm]           # -> [32 lanes, nb, kb, s]
    return np.ascontiguousarray((lo | (hi << 4)).transpose(1, 2, 0, 3)).tobytes()


def export_encoder(encoder_dir):
    text = (encoder_dir / 'model.mil').read_text()
    reader = ms._BlobStorageReader(str(encoder_dir / 'weights/weight.bin'))
    BLOB = r'BLOBFILE\(path = string\("@model_path/weights/weight\.bin"\), offset = uint64\((\d+)\)\)'

    consts = {}
    for m in re.finditer(r'tensor<fp16, \[([0-9, ]*)\]> (\w+) = const\(\)\[name = string\("\w+"\), val = tensor<fp16, \[[0-9, ]*\]>\(' + BLOB, text):
        shape = [int(x) for x in m.group(1).split(',')]
        consts[m.group(2)] = ('f16', reader.read_fp16_data(int(m.group(3))).view(np.float16).astype(np.float32).reshape(shape))
    for m in re.finditer(r'tensor<fp16, \[([0-9, ]*)\]> (\w+) = constexpr_blockwise_shift_scale\(data = tensor<int8, \[([0-9, ]*)\]>\(' + BLOB
                         + r'\), scale = tensor<fp16, \[([0-9, ]*)\]>\(' + BLOB, text):
        shape = [int(x) for x in m.group(1).split(',')]
        sshape = [int(x) for x in m.group(5).split(',')]
        q = reader.read_int8_data(int(m.group(4))).astype(np.float32).reshape(shape)
        s = reader.read_fp16_data(int(m.group(6))).view(np.float16).astype(np.float32).reshape(sshape)
        consts[m.group(2)] = ('f16', q * s)
    for m in re.finditer(r'tensor<fp16, \[([0-9, ]*)\]> (\w+) = constexpr_lut_to_dense\(indices = tensor<uint4, \[([0-9, ]*)\]>\(' + BLOB
                         + r'\), lut = tensor<fp16, \[([0-9, ]*)\]>\(' + BLOB, text):
        shape = [int(x) for x in m.group(1).split(',')]
        idx = reader.read_uint4_data(int(m.group(4))).reshape(shape[0], -1)
        lut = reader.read_fp16_data(int(m.group(6))).view(np.float16).reshape(shape[0], 16)
        consts[m.group(2)] = ('lut', (idx, lut))
    scalars = {m.group(1): m.group(2) for m in re.finditer(r'(\w+) = const\(\)\[name = string\("\w+"\), val = ([^\]]*)\]', text)}

    ops = []
    for line in text.split('\n'):
        m = re.match(r'\s*tensor<[^>]*> (\w+) = (linear|conv|layer_norm|add)\((.*)\)\[name', line)
        if m:
            ops.append((m.group(2), m.group(1), dict(re.findall(r'(\w+) = (\w+)', m.group(3)))))


    def get(name):
        return consts[name]


    tensors = {}


    def put(name, value):
        tensors[name] = value


    def dense(name, kind_value):
        kind, value = kind_value
        assert kind == 'f16', name
        put(name, ('f16', value))


    def lut(name, kind_value, rows=None, cols=None):
        kind, value = kind_value
        assert kind == 'lut', name
        idx, table = value
        if rows is not None:
            idx, table = idx[rows], table[rows]
        if cols is not None:
            idx = idx[:, cols]
        put(name, ('lut', (idx, table)))


    linears = [o for o in ops if o[0] == 'linear']
    convs = [o for o in ops if o[0] == 'conv']
    norms = [o for o in ops if o[0] == 'layer_norm']
    assert len(linears) == 3 + 24 * 11 and len(convs) == 6 + 24 and len(norms) == 1 + 24 * 5, (len(linears), len(convs), len(norms))

    # Subsampling. Activations are kept [time, freq, channel]; the final linear expects [channel, freq] order, so its
    # columns are permuted to [freq, channel].
    for i, key in enumerate(['sub.conv0', 'sub.dw1', 'sub.pw1', 'sub.dw2', 'sub.pw2']):
        w, b = convs[i][2]['weight'], convs[i][2]['bias']
        kind, value = get(w)
        if kind == 'lut':
            lut(key + '.w', (kind, value))
        else:
            dense(key + '.w', ('f16', value.reshape(value.shape[0], -1)))
        dense(key + '.b', get(b))
    perm = np.array([c * 16 + f for f in range(16) for c in range(256)])
    lut('sub.linear.w', get(linears[0][2]['weight']), cols=perm)
    dense('sub.linear.b', get(linears[0][2]['bias']))

    # Insert adapter: LN, down, depthwise 15 + gelu, up, residual.
    dense('ins.ln.g', get(norms[0][2]['gamma'])); dense('ins.ln.b', get(norms[0][2]['beta']))
    lut('ins.down.w', get(linears[1][2]['weight'])); dense('ins.down.b', get(linears[1][2]['bias']))
    dense('ins.mix.w', ('f16', get(convs[5][2]['weight'])[1].reshape(1024, 15))); dense('ins.mix.b', get(convs[5][2]['bias']))
    lut('ins.up.w', get(linears[2][2]['weight'])); dense('ins.up.b', get(linears[2][2]['bias']))

    dense('pos_emb', ('f16', consts['pos_emb_to_fp16'][1].reshape(375, 1024)))

    adds = [o for o in ops if o[0] == 'add' and o[2].get('y', '') in consts and consts[o[2]['y']][1].shape == (1, 1, 8, 128)]
    assert len(adds) == 48, len(adds)
    interleave = np.array([c for pair in zip(range(1024), range(1024, 2048)) for c in pair])
    for layer in range(24):
        L = linears[3 + layer * 11: 3 + (layer + 1) * 11]
        N = norms[1 + layer * 5: 1 + (layer + 1) * 5]
        p = f'l{layer}.'
        for j, key in enumerate(['ln_ff1', 'ln_att', 'ln_conv', 'ln_ff2', 'ln_out']):
            dense(p + key + '.g', get(N[j][2]['gamma'])); dense(p + key + '.b', get(N[j][2]['beta']))
        lut(p + 'ff1a.w', get(L[0][2]['weight'])); dense(p + 'ff1a.b', get(L[0][2]['bias']))
        lut(p + 'ff1b.w', get(L[1][2]['weight'])); dense(p + 'ff1b.b', get(L[1][2]['bias']))
        q, k, v = (get(L[i][2]['weight'])[1] for i in (2, 3, 4))
        put(p + 'qkv.w', ('lut', (np.concatenate([q[0], k[0], v[0]]), np.concatenate([q[1], k[1], v[1]]))))
        put(p + 'qkv.b', ('f16', np.concatenate([get(L[i][2]['bias'])[1] for i in (2, 3, 4)])))
        dense(p + 'bias_u', ('f16', consts[adds[2 * layer][2]['y']][1].reshape(1024)))
        dense(p + 'bias_v', ('f16', consts[adds[2 * layer + 1][2]['y']][1].reshape(1024)))
        lut(p + 'pos.w', get(L[5][2]['weight'])); dense(p + 'pos.b', get(L[5][2]['bias']))
        lut(p + 'o.w', get(L[6][2]['weight'])); dense(p + 'o.b', get(L[6][2]['bias']))
        lut(p + 'pw1.w', get(L[7][2]['weight']), rows=interleave); dense(p + 'pw1.b', ('f16', get(L[7][2]['bias'])[1][interleave]))
        dense(p + 'dw.w', ('f16', get(convs[6 + layer][2]['weight'])[1].reshape(1024, 9))); dense(p + 'dw.b', get(convs[6 + layer][2]['bias']))
        lut(p + 'pw2.w', get(L[8][2]['weight'])); dense(p + 'pw2.b', get(L[8][2]['bias']))
        lut(p + 'ff2a.w', get(L[9][2]['weight'])); dense(p + 'ff2a.b', get(L[9][2]['bias']))
        lut(p + 'ff2b.w', get(L[10][2]['weight'])); dense(p + 'ff2b.b', get(L[10][2]['bias']))

    return tensors


def export_decoder(model_dir):
    def blobs(part, entries):
        reader = ms._BlobStorageReader(str(model_dir / f'{part}.mlmodelc/weights/weight.bin'))
        return {name: reader.read_fp16_data(off).view(np.float16).reshape(shape) for name, (off, shape) in entries.items()}


    t = {}
    t.update(blobs('Decoder', {
        'emb': (64, (8193, 640)),
        'l0.wih': (10487168, (2560, 640)), 'l0.whh': (13764032, (2560, 640)), 'l0.b': (17040896, (2560,)),
        'l1.wih': (17046080, (2560, 640)), 'l1.whh': (20322944, (2560, 640)), 'l1.b': (23599808, (2560,)),
    }))
    t.update(blobs('JointDecision', {
        'enc.w': (64, (640, 1024)), 'enc.b': (1310848, (640,)),
        'dec.w': (1312192, (640, 640)), 'dec.b': (2131456, (640,)),
        'head.w': (2132800, (8198, 640)), 'head.b': (12626304, (8198,)),
    }))
    # One matrix per LSTM layer: gates = [W_ih | W_hh] @ [x; h] + b, gate rows in order i, f, o, g.
    for l in (0, 1):
        t[f'l{l}.w'] = np.concatenate([t.pop(f'l{l}.wih'), t.pop(f'l{l}.whh')], axis=1)
        t[f'l{l}.b'] = t.pop(f'l{l}.b')
    return {name: ('f16', value) for name, value in t.items()}


if __name__ == '__main__':
    encoder_dir, model_dir, out = (Path(a) for a in sys.argv[1:4])
    out.mkdir(parents=True, exist_ok=True)
    write(out / 'encoder.bin', 'swiftlet-encoder-wq2-1', export_encoder(encoder_dir))
    write(out / 'decoder.bin', 'swiftlet-decoder-1', export_decoder(model_dir))
