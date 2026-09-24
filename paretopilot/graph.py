"""Shape-only recommendation graph; no tensors or model weights are allocated."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod

from .config import ConfigError, DTYPE_BYTES


@dataclass
class Tensor:
    name: str
    shape: tuple
    element_bytes: int
    external: bool = False

    @property
    def size_bytes(self):
        return prod(self.shape) * self.element_bytes


@dataclass
class Operator:
    name: str
    kind: str
    inputs: list[str]
    output: str
    useful_ops: int
    parameter_bytes_read: int = 0
    matrix: tuple | None = None  # groups, M, K, N
    lookup_bytes: int = 0


@dataclass
class Graph:
    tensors: dict[str, Tensor] = field(default_factory=dict)
    operators: list[Operator] = field(default_factory=list)
    parameters: dict[str, int] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    element_bytes: int = 2

    @property
    def parameter_count(self):
        return sum(self.parameters.values())

    @property
    def weight_bytes(self):
        return self.parameter_count * self.element_bytes

    def peak_activation_bytes(self):
        """Sequential execution, out-of-place outputs, exact tensor last-use release."""
        uses = {key: 0 for key in self.tensors}
        for op in self.operators:
            for key in op.inputs:
                uses[key] += 1
        for key in self.outputs:
            uses[key] += 1  # returned predictions stay live
        live = {key for key, tensor in self.tensors.items() if tensor.external and uses[key]}
        current = sum(self.tensors[key].size_bytes for key in live)
        peak = current
        for op in self.operators:
            live.add(op.output)
            current += self.tensors[op.output].size_bytes
            peak = max(peak, current)
            for key in op.inputs:
                uses[key] -= 1
                if uses[key] == 0 and key in live:
                    current -= self.tensors[key].size_bytes
                    live.remove(key)
            if uses[op.output] == 0:
                current -= self.tensors[op.output].size_bytes
                live.remove(op.output)
        return peak


class Builder:
    def __init__(self, model, workload):
        self.model, self.workload = model, workload
        self.b = workload["batch_size"]
        self.graph = Graph(element_bytes=DTYPE_BYTES[model["dtype"]])

    def tensor(self, name, shape, external=False, element_bytes=None):
        if name in self.graph.tensors:
            raise ConfigError(f"Duplicate graph tensor: {name}")
        self.graph.tensors[name] = Tensor(name, tuple(shape), element_bytes or self.graph.element_bytes, external)
        return name

    def parameter(self, name, count):
        self.graph.parameters[name] = count
        return count * self.graph.element_bytes

    def op(self, name, kind, inputs, shape, ops=0, param_bytes=0, matrix=None, lookup_bytes=0):
        output = self.tensor(name + ".out", shape)
        self.graph.operators.append(Operator(name, kind, inputs, output, ops, param_bytes, matrix, lookup_bytes))
        return output

    def linear(self, name, x, width):
        shape = self.graph.tensors[x].shape
        rows, inner = prod(shape[:-1]), shape[-1]
        weight = self.parameter(name + ".weight", inner * width)
        return self.op(name, "matmul", [x], (*shape[:-1], width),
                       2 * rows * inner * width, weight, (1, rows, inner, width))

    def elementwise(self, name, kind, x, ops_per_element=1, other=None, param_count=0):
        shape = self.graph.tensors[x].shape
        params = self.parameter(name + ".weight", param_count) if param_count else 0
        return self.op(name, kind, [x] + ([other] if other else []), shape,
                       prod(shape) * ops_per_element, params)

    def mlp(self, name, x, widths):
        for i, width in enumerate(widths):
            x = self.linear(f"{name}.{i}.linear", x, width)
            x = self.elementwise(f"{name}.{i}.relu", "relu", x)
        return x

    def lookup(self, name, vocab, dim, length):
        self.parameter(name + ".table", vocab * dim)
        ids = self.tensor(name + ".ids", (self.b, length), external=True, element_bytes=8)
        return self.op(name, "embedding", [ids], (self.b, length, dim),
                       lookup_bytes=self.b * length * dim * self.graph.element_bytes)

    def pool(self, name, x):
        b, length, dim = self.graph.tensors[x].shape
        return self.op(name, "mean", [x], (b, dim), b * length * dim)

    def concat(self, name, xs):
        if len(xs) == 1:
            return xs[0]
        width = sum(self.graph.tensors[x].shape[-1] for x in xs)
        return self.op(name, "concat", xs, (self.b, width))

    def sequence(self, seq):
        length = self.workload["sequence_length"]
        hidden, heads = seq["hidden_size"], seq["num_heads"]
        head_dim = hidden // heads
        x = self.lookup("sequence.embedding", seq["vocab_size"], seq["embedding_dim"], length)
        if seq["embedding_dim"] != hidden:
            x = self.linear("sequence.input_projection", x, hidden)
        for layer in range(seq["num_layers"]):
            prefix = f"sequence.{layer}"
            q, k, v = [self.linear(prefix + "." + part, x, hidden) for part in ("q", "k", "v")]
            score = self.op(prefix + ".attention_scores", "matmul", [q, k],
                            (self.b, heads, length, length),
                            2 * self.b * heads * length * length * head_dim,
                            matrix=(self.b * heads, length, head_dim, length))
            score = self.elementwise(prefix + ".softmax", "softmax", score, 6)
            context = self.op(prefix + ".attention_context", "matmul", [score, v],
                              (self.b, length, hidden),
                              2 * self.b * heads * length * length * head_dim,
                              matrix=(self.b * heads, length, length, head_dim))
            projected = self.linear(prefix + ".attention_output", context, hidden)
            summed = self.elementwise(prefix + ".residual1", "add", x, other=projected)
            x = self.elementwise(prefix + ".norm1", "layernorm", summed, 7, param_count=2 * hidden)
            ff = self.mlp(prefix + ".ffn", x, [seq["ffn_size"]])
            ff = self.linear(prefix + ".ffn_output", ff, hidden)
            summed = self.elementwise(prefix + ".residual2", "add", x, other=ff)
            x = self.elementwise(prefix + ".norm2", "layernorm", summed, 7, param_count=2 * hidden)
        return self.pool("sequence.pool", x)

    def build(self):
        m = self.model
        features = []
        if m["dense_features"]:
            dense = self.tensor("dense.input", (self.b, m["dense_features"]), external=True)
            features.append(self.mlp("dense", dense, m["dense_mlp"]))
        for index, e in enumerate(m["embeddings"]):
            # Use stable numeric IDs internally; user feature names can contain punctuation.
            prefix = f"embedding.{index}"
            x = self.lookup(prefix, e["vocab_size"], e["dim"], self.workload["lookups_per_sample"][e["name"]])
            features.append(self.pool(prefix + ".pool", x))
        if m["sequence_encoder"]:
            features.append(self.sequence(m["sequence_encoder"]))
        if m["interaction"]["kind"] == "dot":
            if len(features) < 2:
                raise ConfigError("dot interaction requires at least two feature vectors")
            dim = m["interaction"]["projection_dim"]
            projected = [self.linear(f"interaction.project.{i}", x, dim)
                         if self.graph.tensors[x].shape[-1] != dim else x
                         for i, x in enumerate(features)]
            count = len(projected)
            stacked = self.op("interaction.stack", "concat", projected, (self.b, count, dim))
            gram = self.op("interaction.gram", "matmul", [stacked], (self.b, count, count),
                           2 * self.b * count * count * dim, matrix=(self.b, count, dim, count))
            pairs = self.op("interaction.pairs", "gather", [gram], (self.b, count * (count - 1) // 2))
            # DLRM-style dense skip: first projected feature plus off-diagonal dot products.
            shared = self.concat("interaction.output", [projected[0], pairs])
        else:
            shared = self.concat("interaction.concat", features)
        for i, tower in enumerate(m["towers"]):
            x = self.mlp(f"tower.{i}", shared, tower["hidden_sizes"])
            x = self.linear(f"tower.{i}.prediction", x, tower["output_dim"])
            x = self.elementwise(f"tower.{i}.sigmoid", "sigmoid", x, 5)
            self.graph.outputs.append(x)
        return self.graph


def build_graph(model, workload):
    return Builder(model, workload).build()
