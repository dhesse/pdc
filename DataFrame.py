import scipy
from collections import defaultdict
from itertools import product
import logging

def n():
    return lambda mask, _: mask[mask.nonzero()].shape[0]

def summarize_fn(fn):
    def wrapped(*cols):
        def inner(mask, df):
            cols_actual = []
            for col in cols:
                try:
                    cols_actual.append(df[col])
                except TypeError:
                    cols_actual.append(col)
            return fn(*[col[mask.nonzero()] for col in cols_actual])
        return inner
    return wrapped

@summarize_fn
def mean(col):
    return col.mean()

@summarize_fn
def sd(col):
    return col.std()

class Column(object):
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent
    def get(self):
        return self.parent[self.name]
    def __hash__(self):
        return hash(self.name)
    def __str__(self):
        return self.name
    def __add__(self, other):
        return self.get() + other.get()

class DataFrame(object):
    def __init__(self, scope = None, **kwargs):
        self.scope = scope or {}
        self.columns = {}
        self.N = kwargs.values()[0].shape
        for k in kwargs:
            self.register_column(k, kwargs[k])
        self.temps = []
            
    def register_column(self, colname, value):
        self.columns[colname] = value
        if colname not in self.scope:
            self.scope[colname] = Column(colname, self)
        else:
            logging.warn("Variable '{0}' present in globals, not set!".format(colname))
        
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
                vals[str(k)].append(d[k])
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
