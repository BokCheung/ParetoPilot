# Vendored CANN 8.5.0 MatMulV3

Also included: `ops-math-add-v2.zip`, exported with `git archive` from ops-math
tag `v8.5.0`, commit `f1a42c5381bbd33981877ad87751d242663c2ce6`.
`add_manifest.json` records its checksum and experimental/community status.
It contains unchanged root LICENSE/CMake and the common/cmake/scripts/AddV2
subtrees needed for the isolated build. It is not installed stock Add source.
Device files remain unchanged after extraction; integration and Host-control
patches are recorded separately. See `docs/OFFICIAL_ADD.md` in the project.

Upstream: https://gitcode.com/cann/ops-nn, branch `8.5.0`, commit
`9f8a66e795ef1842c2118cf5bbadfe3624bdd1ef`.

`ops-nn-matmul.zip` contains unchanged upstream files: root CMakeLists.txt,
build.sh, version.info, LICENSE, cmake/, scripts/, common/,
matmul/{mat_mul_v3,batch_mat_mul_v3,common}/, and conv/common/op_kernel/.
The latter supplies shared files copied by the upstream build. Tests and
supporting architecture headers in these subtrees are retained. Files were
exported with `git archive` from the pinned commit, not from a working tree.
The archive can be inspected using Python's standard `zipfile` module.

`build-deps.zip` contains nlohmann/json v3.11.3's single header (MIT) and
CANN makeself release-2.5.0-patch1.0 (GPL-2.0-or-later), including licenses.
See manifest.json for hashes, source URLs, versions, and the complete upstream
file inventory. CANN code retains the adjacent CANN Open Software License
Agreement Version 2.0; it is not relicensed under the licenses of dependencies.

No device computation code is patched. `paretopilot/official.py` verifies these
archives, extracts a trial-local copy, and records host/build adaptations.
The build uses the selected machine's licensed CANN development libraries.
No network download or system OPP modification is part of the build.

Build adaptations disable unused generic ONNX/protobuf/Eigen dependencies and
add a GE Operator ONNX parser. Graph integration adds MatMulV3's missing dtype
inference to its existing shape registration: output y takes input x1's dtype,
consistent with the upstream OpDef. The upstream shape callback is unchanged.
Host adaptations expose a core budget and knowledge-bank policy, and print
provenance/effective tiling. Defaults retain the original tiling decisions.
See docs/OFFICIAL_MATMUL.md for limitations.

The Python adapter install rule enumerates dynamic/*.py at install time, after
generation, instead of freezing an empty file list during initial configuration.
Generated adapters are checked for presence and matching hashes in the installed
OPP before ATC runs; build_manifest.json records the verified Python files.
