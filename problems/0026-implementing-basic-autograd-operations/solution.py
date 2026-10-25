class Value:

  def __init__(self, data, _children=(), _op=""):
    self.data = data
    self.grad = 0
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op

  def __repr__(self):

    def fmt(x):
      return int(x) if float(x).is_integer() else round(x, 4)

    return f"Value(data={fmt(self.data)}, grad={fmt(self.grad)})"

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), "+")

    def _backward():
      self.grad += out.grad
      other.grad += out.grad

    out._backward = _backward
    return out

  def __radd__(self, other):
    return self + other

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), "*")

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad

    out._backward = _backward
    return out

  def __rmul__(self, other):
    return self * other

  def relu(self):
    out = Value(self.data if self.data > 0 else 0, (self,), "ReLU")

    def _backward():
      # 当输入大于 0 时传递梯度，否则置为 0
      self.grad += (1 if self.data > 0 else 0) * out.grad

    out._backward = _backward
    return out

  def backward(self):
    # 1. 深度优先搜索建立逆序拓扑序列
    topo = []
    visited = set()

    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)

    build_topo(self)

    # 2. 根节点的初始梯度置为 1
    self.grad = 1

    # 3. 按逆拓扑序依次调用反向求导闭包
    for node in reversed(topo):
      node._backward()