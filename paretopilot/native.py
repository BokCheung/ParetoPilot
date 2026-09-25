"""Lower native recommendation modules to explicit, auditable Roofline kernels.

Custom module names are not specifications. Defaults below are approximation
profiles, overridable by module type or node in the native workload.
"""

from __future__ import annotations

from math import prod

from .config import ConfigError, integer, obj, sizes
from .graph import Builder
from .native_config import Feature, FeatureSet, flatten, normalize_native, resolve
from .roofline import ASSUMPTIONS, evaluate_graph


# F counts useful operations (one MAC = two ops), B=batch, L=sequence length,
# D=width, T=tokens. Traffic is calculated from each lowered primitive's shapes.
PROFILES = {
    "sfps_feature_embedding": ("Input boundary", "Ready tensors only; no lookup, table weights, communication or cache."),
    "log_dense": ("F ~ B*C*bucket_ops; gather B*C*E elements", "Approximate per-feature logarithmic bucketization followed by a local H x E table lookup; not SFPS."),
    "feature_sequential": ("sum: F = B*D*(L-1); mean: F = B*D*L per feature", "Apply declared feature methods to ready tensors; unmatched features pass through."),
    "sum": ("F = output_elements*(reduction_length-1)", "Out-of-place reduction, retaining only declared axes."),
    "mean": ("F = output_elements*reduction_length", "Sum reduction followed by scaling."),
    "pooling": ("sum: B*D*(L-1); mean: B*D*L", "Model-side pooling of ready tensors."),
    "split": ("F = 0; bytes = 0 for views", "Zero-copy slices; parent storage stays live until every slice is released."),
    "squeeze": ("F = 0; bytes = 0", "Metadata-only removal of size-one axes."),
    "flatten_list": ("F = 0; bytes = 0", "Flatten Python containers, not tensor dimensions; no device kernel."),
    "concatenate": ("F = 0; bytes = sum(input_bytes)+output_bytes", "Materialized concatenation; a single input is a pass-through."),
    "stack": ("F = 0; bytes = sum(input_bytes)+output_bytes", "Materialized stack of equal shapes."),
    "dense": ("F_matrix = 2*M*K*N; P = K*N (+N bias)", "Dense matrix multiply with optional separate bias and declared activation."),
    "dnn": ("F_matrix = sum(2*M*d_i*d_(i+1))", "MLP; default hidden activation ReLU, bias enabled."),
    "loop": ("F = sum(branch primitive F)", "Independent parameters in each layer_seq branch; supports concat and dense stages."),
    "din": ("F_matrix = sum_pairs sum_layers 2*B*L*d_i*d_(i+1); d_0=4D", "Local attention MLP on [q,k,q-k,q*k], then unnormalized weighted sum; DICE uses inference statistics."),
    "din_v2": ("Same DIN formula for the supplied target/sequence pair", "Same approximation as DIN; no undocumented v2 behavior inferred."),
    "ads_can": ("F_matrix = sum_pairs 2*B*L*sum_layers(d_i*d_(i+1))", "Co-action approximation: sequence embeddings encode per-example MLP matrices, target is the MLP input; no additional learned matrices. Default 24 = 8 x 3, tanh, one order, no bias."),
    "epnet": ("Gate MLP + 2*B*D for scale and multiply", "Approximation: concat(domain, features) -> gate MLP -> 2*sigmoid -> multiply features. stop_gradient has no inference cost."),
    "gatenu": ("Gate MLP + 2*B*D for scale and multiply", "Approximation: conditioning input -> gate MLP -> 2*sigmoid -> multiply feature input."),
    "token_mixer": ("F_matrix = 2*B*D*T*T by default", "Approximation: shared learned T -> T linear mixing along tokens, with materialized transposes; configurable hidden widths."),
    "per_token_ffn": ("F_matrix = 4*B*T*D*H; P_matrix = 2*T*D*H", "Approximation: independent D -> H -> D MLP per token, default H=4D, ReLU, then residual and layer norm as configured. Inference dropout is identity."),
    "hr_tower": ("F_matrix = 2*B*((D+C)*H + H*O)", "Proxy only: concat(feature, domain) -> H -> O MLP. H=hr_layer_dim, O=1 by default; true HR routing is unknown."),
    "string_expression": ("F = 4*output_elements for x/(x+(1-x)/y)", "Only the specified negative-sampling correction is recognized; scalar y is a compile-time constant."),
}
CUSTOM = {"log_dense", "din", "din_v2", "ads_can", "epnet", "gatenu", "token_mixer", "per_token_ffn", "hr_tower"}
OPTION_KEYS = {
    "log_dense": {"bucket_ops"}, "dense": {"use_bias"}, "dnn": {"use_bias"},
    "din": {"use_bias", "normalize_weights"}, "din_v2": {"use_bias", "normalize_weights"},
    "ads_can": {"hidden_sizes", "activation"},
    "epnet": {"hidden_sizes", "use_bias"}, "gatenu": {"hidden_sizes", "use_bias"},
    "token_mixer": {"hidden_sizes", "use_bias"},
    "per_token_ffn": {"hidden_size", "shared_weights", "use_bias"},
    "hr_tower": {"output_dim", "use_bias"}, "loop": {"use_bias"},
}


class NativeBuilder(Builder):
    def __init__(self, model, workload):
        super().__init__(model, workload)
        self.modules = []
        self.environment = {}
        self.ready = self.features("ready_features", "discrete")
        self.environment["inputs"] = self.features("dense_features", "continuous")
        unknown = set(workload["module_options"]) - set(PROFILES)
        if unknown:
            raise ConfigError(f"Unknown module_options: {sorted(unknown)}")

    def features(self, field, feature_type):
        result = {}
        for item in self.workload[field]:
            tensor = self.tensor(f"{field}.{item['name']}", (self.b, *item["shape"]), external=True)
            result[item["name"]] = Feature(tensor, {"name": item["name"], "type": feature_type,
                                                   "region": item["region"], "kind": item.get("kind")})
        return FeatureSet(result)

    def shape(self, x):
        if not isinstance(x, str) or x not in self.graph.tensors:
            raise ConfigError("Expected one tensor")
        return self.graph.tensors[x].shape

    def parameter(self, name, count):
        if name in self.graph.parameters and self.graph.parameters[name] != count:
            raise ConfigError(f"Shared parameter shape mismatch: {name}")
        return super().parameter(name, count)

    def view(self, name, x, shape, slice_view=False):
        size = prod(shape)
        if (not slice_view and size != prod(self.shape(x))) or size > prod(self.shape(x)):
            raise ConfigError(f"{name}: invalid view size")
        out = self.op(name, "view", [x], shape)
        self.graph.tensors[out].alias_of = x
        return out

    def activation(self, name, x, activation, shared=None):
        if activation in (None, "linear", "identity"):
            return x
        costs = {"relu": 1, "sigmoid": 5, "tanh": 6, "dice": 11}
        if activation not in costs:
            raise ConfigError(f"Unknown activation: {activation}")
        params = 3 * self.shape(x)[-1] if activation == "dice" else 0
        param_bytes = self.parameter((shared or name) + ".state", params) if params else 0
        return self.op(name, activation, [x], self.shape(x), prod(self.shape(x))*costs[activation], param_bytes)

    def dense(self, name, x, width, activation=None, bias=True, shared=None):
        integer(width, name + ".units")
        shape = self.shape(x)
        m, k = prod(shape[:-1]), shape[-1]
        parameter_name = shared or name
        weight = self.parameter(parameter_name + ".weight", k * width)
        x = self.op(name + ".matmul", "matmul", [x], (*shape[:-1], width), 2*m*k*width,
                    weight, (1, m, k, width))
        if bias:
            x = self.op(name + ".bias", "add", [x], self.shape(x), m * width,
                        self.parameter(parameter_name + ".bias", width))
        return self.activation(name + ".activation", x, activation, parameter_name + ".activation")

    def network(self, name, x, widths, activation="relu", bias=True, last_activation=None, shared=None):
        sizes(widths, name + ".hidden_sizes")
        for i, width in enumerate(widths):
            act = last_activation if i == len(widths) - 1 else activation
            x = self.dense(f"{name}.{i}", x, width, act, bias, f"{shared}.{i}" if shared else None)
        return x

    def concatenate(self, name, value, axis=-1, stack=False):
        xs = flatten(value)
        if not xs:
            raise ConfigError(f"{name}: cannot concatenate an empty feature selection")
        shapes = [self.shape(x) for x in xs]
        rank = len(shapes[0]) + int(stack)
        if type(axis) is not int or not -rank <= axis < rank:
            raise ConfigError(f"{name}: axis out of range")
        axis %= rank
        if stack:
            if any(shape != shapes[0] for shape in shapes):
                raise ConfigError(f"{name}: stack inputs must have equal shapes")
            out_shape = list(shapes[0])
            out_shape.insert(axis, len(xs))
        else:
            if any(len(s) != rank or any(s[i] != shapes[0][i] for i in range(rank) if i != axis) for s in shapes):
                raise ConfigError(f"{name}: incompatible concat shapes {shapes}")
            if len(xs) == 1:
                return xs[0]
            out_shape = list(shapes[0])
            out_shape[axis] = sum(s[axis] for s in shapes)
        return self.op(name, "concat", xs, out_shape)

    def reduce(self, name, x, kind, axis=1, keepdims=False):
        shape = list(self.shape(x))
        if type(axis) is not int or not -len(shape) <= axis < len(shape):
            raise ConfigError(f"{name}: reduction axis out of range")
        axis %= len(shape)
        length = shape[axis]
        if keepdims:
            shape[axis] = 1
        else:
            shape.pop(axis)
        ops = prod(shape) * (length if kind == "mean" else length - 1)
        return self.op(name, kind, [x], shape, ops)

    def attention(self, name, target, sequence, parameters, options, shared=None):
        qshape, kshape = self.shape(target), self.shape(sequence)
        if len(qshape) == 3 and qshape[1] == 1:
            target = self.view(name + ".target_view", target, (qshape[0], qshape[2]))
            qshape = self.shape(target)
        if len(qshape) != 2 or len(kshape) != 3 or qshape != (kshape[0], kshape[2]):
            raise ConfigError(f"{name}: DIN requires target [B,D] or [B,1,D], sequence [B,L,D], got {qshape}, {kshape}")
        b, length, dim = kshape
        q = self.op(name + ".broadcast", "gather", [target], kshape)
        diff = self.op(name + ".difference", "subtract", [q, sequence], kshape, b*length*dim)
        product = self.op(name + ".product", "multiply", [q, sequence], kshape, b*length*dim)
        x = self.concatenate(name + ".features", [q, sequence, diff, product])
        config = parameters.get("dnn_config", {})
        if isinstance(config, list):
            if len(config) != 1:
                raise ConfigError("DIN supports one dnn_config definition")
            config = config[0]
        widths = config.get("hidden_dims", [32, 1])
        if not widths or widths[-1] != 1:
            raise ConfigError("DIN scoring MLP must end in width 1")
        activation = config.get("hidden_activation", ["dice"])
        if isinstance(activation, list):
            if len(activation) != 1:
                raise ConfigError("DIN supports one hidden activation")
            activation = activation[0]
        score = self.network(name + ".score", x, widths, activation, options.get("use_bias", True), shared=shared)
        if options.get("normalize_weights", False):
            score = self.elementwise(name + ".softmax", "softmax", score, 6)
        weighted = self.op(name + ".weighted", "multiply", [sequence, score], kshape, b*length*dim)
        return self.reduce(name + ".pool", weighted, "sum")

    def lower(self, name, kind, p, inputs, options):
        x = inputs.get("inputs")
        bias = options.get("use_bias", p.get("use_bias", True))
        if kind == "sfps_feature_embedding":
            return self.ready
        if kind == "log_dense":
            if not isinstance(x, FeatureSet):
                raise ConfigError("log_dense requires continuous feature metadata")
            width, buckets = p["embedding_size"], p["hash_num"]
            integer(width, "log_dense.embedding_size")
            integer(buckets, "log_dense.hash_num")
            result = {}
            for i, (key, feature) in enumerate(x.features.items()):
                tensor, prefix = feature.tensor, f"{name}.{i}"
                # This aggregate represents log/abs/scale/clamp/floor; the count
                # is an explicit approximate vector cost, not measured latency.
                indices = self.op(prefix + ".bucket", "log", [tensor], self.shape(tensor),
                                  prod(self.shape(tensor)) * options.get("bucket_ops", 5))
                self.graph.tensors[indices].element_bytes = 4
                self.parameter(prefix + ".table", buckets * width)
                out_shape = (*self.shape(tensor), width)
                embedding = self.op(prefix + ".lookup", "gather", [indices], out_shape,
                                    param_bytes=prod(out_shape)*self.graph.element_bytes)
                result[key] = Feature(embedding, feature.metadata)
            return [FeatureSet(result)]
        if kind == "feature_sequential":
            if not isinstance(x, FeatureSet):
                raise ConfigError("feature_sequential requires feature metadata")
            result = {}
            for i, (key, feature) in enumerate(x.features.items()):
                tensor = feature.tensor
                methods = p.get("methods", {})
                selector = "features#" + feature.metadata["kind"]
                for j, method in enumerate(methods.get(selector, [])):
                    operation = method.get("type")
                    if operation not in ("sum", "mean"):
                        raise ConfigError(f"Unsupported feature method: {operation}")
                    tensor = self.reduce(f"{name}.{i}.{j}", tensor, operation, method.get("axis", 1), method.get("keepdims", False))
                result[key] = Feature(tensor, feature.metadata)
            unknown = set(p.get("methods", {})) - {"features#single_discrete", "features#multiple_discrete"}
            if unknown:
                raise ConfigError(f"Unsupported feature selectors: {sorted(unknown)}")
            return FeatureSet(result)
        if kind in ("sum", "mean", "pooling"):
            operation = p.get("mode", "mean") if kind == "pooling" else kind
            if operation not in ("sum", "mean"):
                raise ConfigError("pooling.mode must be sum or mean")
            return self.reduce(name, x, operation, p.get("axis", inputs.get("axis", 1)), p.get("keepdims", False))
        if kind == "split":
            tensor = inputs["value"]
            shape = list(self.shape(tensor))
            axis = inputs.get("axis", p.get("axis", -1))
            if type(axis) is not int or not -len(shape) <= axis < len(shape):
                raise ConfigError("split axis out of range")
            axis %= len(shape)
            widths = inputs.get("num_or_size_splits", p.get("num_or_size_splits"))
            if type(widths) is int:
                integer(widths, "split count")
                if shape[axis] % widths:
                    raise ConfigError("split count must divide dimension")
                widths = [shape[axis] // widths] * widths
            sizes(widths, "split widths")
            if sum(widths) != shape[axis]:
                raise ConfigError("split widths do not sum to input dimension")
            outputs = []
            for i, width in enumerate(widths):
                shape[axis] = width
                outputs.append(self.view(f"{name}.{i}", tensor, shape, slice_view=True))
            return outputs
        if kind == "squeeze":
            tensor = inputs.get("input", x)
            shape = list(self.shape(tensor))
            axis = inputs.get("axis", p.get("axis"))
            if axis is None:
                shape = [d for d in shape if d != 1]
            else:
                if type(axis) is not int or not -len(shape) <= axis < len(shape) or shape[axis] != 1:
                    raise ConfigError("squeeze requires a size-one axis")
                shape.pop(axis)
            return self.view(name, tensor, shape)
        if kind == "flatten_list":
            return flatten(x)
        if kind in ("concatenate", "stack"):
            return self.concatenate(name, inputs.get("values", x), inputs.get("axis", p.get("axis", -1)), kind == "stack")
        if kind == "dense":
            return self.dense(name, x, p["units"], p.get("activation"), bias)
        if kind == "dnn":
            activation = p.get("hidden_activation", "relu")
            if isinstance(activation, list):
                if len(activation) != 1:
                    raise ConfigError("dnn supports one hidden activation")
                activation = activation[0]
            return self.network(name, x, p["hidden_dims"], activation, bias, last_activation=activation)
        if kind == "loop":
            loop = p["loop"]
            if loop.get("type") != "layer_seq" or not isinstance(x, list):
                raise ConfigError("loop requires layer_seq and a list of branches")
            result = []
            for i, branch in enumerate(x):
                for j, stage in enumerate(loop["config_list"]):
                    if stage.get("type") not in ("concatenate", "dense"):
                        raise ConfigError("loop supports concatenate/dense stages")
                    branch = self.lower(f"{name}.{i}.{j}", stage["type"], stage, {"inputs": branch}, options)
                result.append(branch)
            return result
        if kind in ("din", "din_v2"):
            specific = p.get("din_specific_params", p)
            if kind == "din":
                if not isinstance(x, FeatureSet):
                    raise ConfigError("DIN requires a feature selection")
                targets, sequences = specific["target_feature"], specific["sequence_features"]
                if len(targets) != len(sequences) or not targets:
                    raise ConfigError("DIN target/sequence feature lists must match")
                if any(not isinstance(group, list) or len(group) != 1 for group in sequences):
                    raise ConfigError("DIN currently requires one sequence feature per target")
                pairs = [(x.features[q].tensor, x.features[group[0]].tensor) for q, group in zip(targets, sequences)]
            else:
                pairs = [tuple(flatten(x))]
                if len(pairs[0]) != 2:
                    raise ConfigError("DIN v2 requires target and sequence")
            share = name + ".shared_score" if p.get("use_same_dnn", False) else None
            return [self.attention(f"{name}.{i}", q, k, specific, options, share) for i, (q, k) in enumerate(pairs)]
        if kind == "ads_can":
            if not isinstance(x, list) or len(x) != 2:
                raise ConfigError("CAN expects [sequence_weight_tensors, target_tensors]")
            sequences, targets = flatten(x[0]), flatten(x[1])
            initial = p["can_target_embedding_size"]
            encoded = p["can_sequence_embedding_size"]
            integer(initial, "CAN target embedding size")
            integer(encoded, "CAN sequence embedding size")
            widths = options.get("hidden_sizes", [encoded // initial])
            sizes(widths, "CAN hidden_sizes")
            if not widths or sum(a*b for a, b in zip([initial, *widths], widths)) != encoded:
                raise ConfigError("CAN embedding width must equal sum of bias-free dynamic matrix sizes; configure hidden_sizes")
            results = []
            for i, sequence in enumerate(sequences):
                ss = self.shape(sequence)
                if len(ss) not in (2, 3) or ss[-1] != encoded:
                    raise ConfigError("CAN weight input must be [B,W] or [B,L,W]")
                length = ss[1] if len(ss) == 3 else 1
                for j, target in enumerate(targets):
                    if self.shape(target) != (self.b, initial):
                        raise ConfigError("CAN target must be [B,can_target_embedding_size]")
                    hidden, previous = target, initial
                    prefix = f"{name}.{i}.{j}"
                    for layer, width in enumerate(widths):
                        weight = self.view(f"{prefix}.{layer}.weight", sequence,
                                           (self.b, length, previous, width), slice_view=True)
                        hidden = self.op(f"{prefix}.{layer}.matmul", "matmul", [hidden, weight],
                                         (self.b, length, width), 2*self.b*length*previous*width,
                                         matrix=(self.b*length, 1, previous, width))
                        hidden = self.activation(f"{prefix}.{layer}.activation", hidden, options.get("activation", "tanh"))
                        previous = width
                    results.append(self.reduce(prefix + ".pool", hidden, "sum"))
            return self.concatenate(name + ".concat", results)
        if kind in ("epnet", "gatenu"):
            tensors = flatten(x)
            if len(tensors) != 2:
                raise ConfigError("Gate requires conditioning and feature tensors")
            condition, feature = tensors
            if len(self.shape(condition)) != 2 or len(self.shape(feature)) != 2:
                raise ConfigError("Gate requires [B,D] tensors")
            if kind == "epnet":
                condition = self.concatenate(name + ".condition", [condition, feature])
            gate = self.network(name + ".gate", condition, [*options.get("hidden_sizes", [64]), self.shape(feature)[-1]],
                                p.get("activation", "relu"), bias, last_activation="sigmoid")
            gate = self.elementwise(name + ".scale2", "multiply", gate)
            return self.op(name + ".apply", "multiply", [gate, feature], self.shape(feature), prod(self.shape(feature)))
        if kind in ("token_mixer", "per_token_ffn"):
            shape = self.shape(x)
            tokens, dim = p["num_tokens"], p["d_model"]
            integer(tokens, kind + ".num_tokens")
            integer(dim, kind + ".d_model")
            if shape != (self.b, tokens, dim):
                raise ConfigError(f"{kind}: expected [B,{tokens},{dim}], got {shape}")
            if kind == "token_mixer":
                transposed = self.op(name + ".transpose_in", "gather", [x], (self.b, dim, tokens))
                mixed = self.network(name + ".token_mlp", transposed, [*options.get("hidden_sizes", []), tokens], bias=bias)
                return self.op(name + ".transpose_out", "gather", [mixed], shape)
            hidden = options.get("hidden_size", 4*dim)
            integer(hidden, "per_token_ffn.hidden_size")
            groups = 1 if options.get("shared_weights", False) else tokens
            rows = self.b*tokens if groups == 1 else self.b
            current = x
            for i, (inner, width) in enumerate(((dim, hidden), (hidden, dim))):
                weight = self.parameter(f"{name}.{i}.weight", groups*inner*width)
                current = self.op(f"{name}.{i}.matmul", "matmul", [current], (self.b, tokens, width),
                                  2*self.b*tokens*inner*width, weight, (groups, rows, inner, width))
                if bias:
                    current = self.op(f"{name}.{i}.bias", "add", [current], self.shape(current), self.b*tokens*width,
                                      self.parameter(f"{name}.{i}.bias", groups*width))
                if i == 0:
                    current = self.activation(name + ".relu", current, "relu")
            if p.get("use_residual", False):
                current = self.elementwise(name + ".residual", "add", current, other=x)
            if p.get("use_norm", False):
                current = self.elementwise(name + ".norm", "layernorm", current, 7, param_count=2*dim)
            return current
        if kind == "hr_tower":
            domain = flatten(inputs["domain_feature_list"])
            joined = self.concatenate(name + ".condition", [x, domain])
            return self.network(name + ".proxy", joined, [p["hr_layer_dim"], options.get("output_dim", 1)], bias=bias)
        if kind == "string_expression":
            if p.get("expression", "").replace(" ", "") != "x/(x+(1-x)/y)":
                raise ConfigError("Only negative-sampling expression x/(x+(1-x)/y) is supported")
            if not isinstance(x, dict) or set(x) != {"x", "y"} or type(x["y"]) not in (float, int) or x["y"] <= 0:
                raise ConfigError("Negative sampling requires tensor x and positive constant y")
            tensor = x["x"]
            inv = self.elementwise(name + ".one_minus", "subtract", tensor)
            scaled = self.elementwise(name + ".divide_y", "divide", inv)
            denominator = self.elementwise(name + ".denominator", "add", tensor, other=scaled)
            return self.elementwise(name + ".divide", "divide", tensor, other=denominator)
        raise ConfigError(f"Unsupported native module type: {kind}")

    def build(self):
        for node in self.model["nodes"]:
            name, kind = node["name"], node["type"]
            if kind not in PROFILES:
                raise ConfigError(f"{name}: unsupported native module {kind}")
            options = {**self.workload["module_options"].get(kind, {}), **self.workload["node_options"].get(name, {})}
            obj(options, OPTION_KEYS.get(kind, set()), set(), name + " options")
            for key in ("use_bias", "use_same_dnn", "use_norm", "use_residual"):
                if key in node.get("parameters", {}) and type(node["parameters"][key]) is not bool:
                    raise ConfigError(f"{name}.{key}: expected boolean")
            for key, value in options.items():
                if key in ("use_bias", "shared_weights", "normalize_weights") and type(value) is not bool:
                    raise ConfigError(f"{name}.{key}: expected boolean")
                if key in ("hidden_size", "output_dim", "bucket_ops"):
                    integer(value, name + "." + key)
                if key == "hidden_sizes":
                    sizes(value, name + ".hidden_sizes")
            first = len(self.graph.operators)
            parameters_before = set(self.graph.parameters)
            try:
                inputs = {} if kind == "sfps_feature_embedding" else resolve(node["inputs"], self.environment)
                output = self.lower(name, kind, node.get("parameters", {}), inputs, options)
            except (KeyError, TypeError, IndexError, ZeroDivisionError) as exc:
                raise ConfigError(f"{name} ({kind}): missing/incompatible module configuration: {exc}") from exc
            except ConfigError as exc:
                raise ConfigError(f"{name} ({kind}): {exc}") from exc
            self.environment[name] = {node["outputs"]: output}
            self.modules.append({"name": name, "type": kind,
                                 "status": "excluded" if kind == "sfps_feature_embedding" else "modeled",
                                 "confidence": "approximation" if kind in CUSTOM else "declared_operations",
                                 "formula": PROFILES[kind][0], "assumption": PROFILES[kind][1], "options": options,
                                 "output_shapes": [list(self.shape(t)) for t in flatten(output)],
                                 "operator_names": [op.name for op in self.graph.operators[first:]],
                                 "parameter_count": sum(self.graph.parameters[k] for k in set(self.graph.parameters)-parameters_before)})
        outputs = self.workload.get("outputs", [f"ref::{self.model['nodes'][-1]['name']}.{self.model['nodes'][-1]['outputs']}"])
        self.graph.outputs = flatten([resolve(ref, self.environment) for ref in outputs])
        return self.graph


def evaluate_native(model, npu, workload):
    model, npu, workload = normalize_native(model, npu, workload)
    builder = NativeBuilder(model, workload)
    graph = builder.build()
    assumptions = ASSUMPTIONS[:7] + ASSUMPTIONS[-3:] + [
        "Native custom module definitions are approximation profiles, not recovered implementations; see each module's assumption.",
        "Native dense biases are included by default. DICE inference counts 11 ops/element and three stored statistics/parameters per channel.",
        "Split/squeeze are zero-copy views with storage-aware liveness; flatten_list is container-only. Transposes/concat/stack are materialized.",
        "log_dense local bucket tables are model-internal; full capacity is counted but only selected entries are read.",
        "Every declared native node executes once in file order, including branches not used by the final output; no compiler dead-code elimination.",
    ]
    if workload.get("synthetic", False):
        assumptions.append("SYNTHETIC WORKLOAD: feature counts, regions and sequence lengths are illustrative, not production measurements.")
    result = evaluate_graph(graph, model, npu, workload, assumptions)
    result["workload"] = workload
    result["estimate_kind"] = "approximate_native_modules" + ("_synthetic_workload" if workload.get("synthetic", False) else "")
    by_name = {op["name"]: op for op in result["operators"]}
    for module in builder.modules:
        ops = [by_name[name] for name in module["operator_names"]]
        module["operators"] = ops
        module["metrics"] = {key: sum(op[key] for op in ops) for key in
                             ("useful_ops", "executed_ops", "external_bytes", "compute_time_ms", "external_memory_time_ms", "latency_ms")}
        metrics = module["metrics"]
        metrics["arithmetic_intensity"] = metrics["executed_ops"] / metrics["external_bytes"] if metrics["external_bytes"] else None
        module["bottleneck"] = "mixed" if len({op["bottleneck"] for op in ops if op["kind"] != "view"}) > 1 else next((op["bottleneck"] for op in ops if op["kind"] != "view"), "no_device_work")
    result["modules"] = builder.modules
    return result
