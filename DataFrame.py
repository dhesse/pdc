import scipy
from collections import defaultdict
from itertools import product

def n():
    return lambda mask, x: mask[mask.nonzero()].shape[0]

class DataFrame(object):
    def __init__(self, **kwargs):
        self.columns = {}
        self.N = kwargs.values()[0].shape
        for k in kwargs:
            self.columns[k] = kwargs[k]
    def group_by(self, *args):
        self.group_vecs = [(scipy.ones(self.N, bool), {})]
        for col in args:
            self.group_vecs = [(v & (self[col] == val),
                               dict(d, **{col: val}))
                               for (v, d), val in product(self.group_vecs,
                                                          scipy.unique(self[col]))]
        self.group_vecs = [(m, d) for m, d in self.group_vecs
                           if m.any()]
        return self
    def summarize(self, **kwargs):
        vals = defaultdict(list)
        for _, d in self.group_vecs:
            for k in d:
                vals[k].append(d[k])
        for k in vals:
            vals[k] = scipy.array(vals[k])
        for col in kwargs:
            fn = kwargs[col]
            vals[col] = scipy.array(
                [fn(bitmask, self) for bitmask, _ in
                 self.group_vecs])
        return DataFrame(**vals)
    def __getattr__(self, name):
        return self.columns.get(name, None)
    def __getitem__(self, name):
        return self.columns.get(name, None)
    def __setitem__(self, key, value):
        self.columns[key] = value
