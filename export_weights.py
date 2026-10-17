"""
Convert a PyTorch checkpoint (tatt.pt) into a plain NumPy archive, WITHOUT needing PyTorch.
    python export_weights.py results/tatt.pt model/tatt_weights.npz
Used so that the deployed dashboard needs only numpy/pandas/plotly/streamlit.
"""
import io, pickle, sys, zipfile
import numpy as np

DT = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "LongStorage": np.int64, "IntStorage": np.int32, "HalfStorage": np.float16, "BoolStorage": np.bool_}


class _Storage:
    def __init__(self, name): self.name = name


def _rebuild_tensor_v2(storage, offset, size, stride, requires_grad=False, hooks=None, metadata=None):
    arr = storage[offset:]
    return np.lib.stride_tricks.as_strided(arr, shape=tuple(size), strides=tuple(s * arr.itemsize for s in stride)).copy()


class _U(pickle.Unpickler):
    def __init__(self, f, z, prefix):
        super().__init__(f); self.z, self.prefix = z, prefix

    def find_class(self, module, name):
        if name.endswith("Storage"): return _Storage(name)
        if name == "_rebuild_tensor_v2": return _rebuild_tensor_v2
        if name == "_rebuild_parameter": return lambda data, requires_grad, hooks, *a: data
        if module == "collections" and name == "OrderedDict":
            import collections; return collections.OrderedDict
        return super().find_class(module, name)

    def persistent_load(self, pid):
        _, stype, key, _loc, _n = pid
        raw = self.z.read(f"{self.prefix}/data/{key}")
        return np.frombuffer(raw, dtype=DT[stype.name]).copy()


def load_checkpoint(path):
    z = zipfile.ZipFile(path)
    pkl = [n for n in z.namelist() if n.endswith("data.pkl")][0]
    prefix = pkl.rsplit("/", 1)[0]
    return _U(io.BytesIO(z.read(pkl)), z, prefix).load()


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    ck = load_checkpoint(src)
    st = ck["state"]; cfg = ck["cfg"]
    np.savez(dst, **{k: np.asarray(v) for k, v in st.items()}, __cfg_json=np.array(__import__("json").dumps({**cfg, "max_len": ck.get("max_len", 256)})))
    print("saved", dst, len(st), "tensors"); [print(" ", k, np.asarray(v).shape) for k, v in st.items()]
