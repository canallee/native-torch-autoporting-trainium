"""Decode a neuronx-cc HLO module proto and surface the ops most likely to trip
compiler lowering (NCC_ITIN902 / NCC_EVRF*) — iota / gather / dynamic-slice /
scatter / pad / sort / complex — with their PyTorch source metadata.

Needs XLA_IR_DEBUG=1 at trace time for source_file/source_line to be populated.

Usage: python analyze_hlo.py /path/to/model.MODULE_xxx.hlo_module.pb

Model-agnostic HLO op inspector for Phase B R5.
"""
import sys, collections
from torch_neuronx.pyhlo import hlo_pb2

# Ops that commonly fail to lower or lower inefficiently on Trainium.
SUSPECT = {"iota", "gather", "scatter", "dynamic-slice", "dynamic-update-slice",
           "pad", "sort", "concatenate", "reduce-window", "select-and-scatter",
           "complex", "fft", "real", "imag"}


def shp(s):
    try:
        return "x".join(str(d) for d in s.dimensions)
    except Exception:
        return "?"


def main(path):
    m = hlo_pb2.HloModuleProto()
    m.ParseFromString(open(path, "rb").read())
    print(f"module: {m.name}")
    opcount = collections.Counter()
    suspects = []
    for comp in m.computations:
        for ins in comp.instructions:
            op = ins.opcode
            opcount[op] += 1
            if op in SUSPECT:
                md = ins.metadata
                suspects.append((op, shp(ins.shape), md.op_type, md.source_file, md.source_line, ins.name))
    print("\n=== opcode histogram (top 30) ===")
    for op, n in opcount.most_common(30):
        print(f"  {n:5d}  {op}")
    print(f"\n=== suspect ops ({len(suspects)}) ===")
    grp = collections.Counter()
    detail = {}
    for op, shape, optype, sf, sl, name in suspects:
        key = (op, sf, sl, optype)
        grp[key] += 1
        detail.setdefault(key, []).append((shape, name))
    for (op, sf, sl, optype), n in grp.most_common():
        ex = detail[(op, sf, sl, optype)][0]
        print(f"  {op:22s} x{n:<4d} shape~{ex[0]:20s} optype={optype:18s} {sf}:{sl}")


if __name__ == "__main__":
    main(sys.argv[1])
