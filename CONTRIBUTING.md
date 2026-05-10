# Contributing to STOK-Core

Thank you for your interest in contributing to STOK-Core, the reference implementation of Option Kernel Bellman Equations!

---

## 🎯 Project Status

STOK-Core implements:
> Ringstrom, T., & Schrater, P. (2025). *A Unified Theory of Compositionality, Modularity, and Interpretability in Markov Decision Processes.* arXiv:2506.09499 — local PDF in `docs/references/ringstomcompositionality.pdf`.

**For active development scope and current parity status, see [`docs/dev/PAPER_PDF_TASKBOARD.md`](docs/dev/PAPER_PDF_TASKBOARD.md).** That board is the only authoritative source for what is done versus open. Older "100/100" / "victory" / "score" docs (now under [`docs/archive/legacy_status_2026/`](docs/archive/legacy_status_2026/)) reflect a self-assessment that pre-dated the formal traceability audit and should NOT be used as implementation authority.

The companion docs under `docs/dev/PAPER_*.md` (traceability matrix, paper-vs-repo assumptions, acceptance checklist, API audit) are the agent-facing detail behind each task board entry.

---

## 🤝 How to Contribute

### 1. Report Issues

Found a bug? Have a question?
- Open an [issue](https://github.com/your-repo/stok-core/issues)
- Include: OS, Rust version, GPU backend, minimal reproduction

### 2. Suggest Enhancements

Interested in extending the framework?
- Check [EXTENSION_DISCUSSION.md](EXTENSION_DISCUSSION.md) for planned extensions
- Open an issue to discuss before implementing
- Major extensions (empowerment, sparse tensors, STOK-RT) should be discussed first

### 3. Submit Pull Requests

**Process**:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. **Run tests**: `cargo test` (must pass 100%)
5. **Run formatter**: `cargo fmt`
6. **Run linter**: `cargo clippy`
7. Commit with descriptive messages
8. Push and create a pull request

**PR Guidelines**:
- Keep changes focused (one feature/fix per PR)
- Include tests for new features
- Update documentation as needed
- Maintain 100% test pass rate

---

## 🧪 Development Guidelines

### Code Quality Standards

**All code must**:
- ✅ Pass all existing tests (`cargo test`)
- ✅ Pass clippy without warnings (`cargo clippy`)
- ✅ Be formatted (`cargo fmt`)
- ✅ Include rustdoc comments for public APIs
- ✅ Reference paper equations where applicable

### Mathematical Correctness

**This is research code!** Correctness is paramount:
- All equation implementations must match the paper exactly
- Include equation numbers in comments (e.g., `// Equation [7]: κ-OKBE`)
- Validate invariants (normalization, bounds, consistency)
- Add tests for any mathematical claims

### Testing Requirements

**New features must include**:
- Unit tests (in the same file under `#[cfg(test)]`)
- Integration tests if feature spans multiple modules
- Property-based tests for mathematical properties
- Examples demonstrating usage

**Current standard**: 220+ tests, 100% passing. Don't break it!

---

## 🎯 Contribution Areas

### Welcome Contributions

**Additional Examples**:
- More application domains (robotics, games, control)
- Different product-space configurations
- Benchmark comparisons

**Performance**:
- Sparse tensor backends
- GPU kernel optimizations
- Memory profiling and improvements

**Documentation**:
- Tutorials and guides
- API usage examples
- Conceptual explanations

**Tooling**:
- Visualization tools (κ heatmaps, STOK distributions)
- Debugging utilities
- Profiling tools

### Major Extensions (Discuss First!)

**These require significant design discussion**:
- Empowerment implementation (requires thesis study)
- Real-time runtime (STOK-RT)
- Continuous state-spaces
- Neural network integration
- Product-space tree search extension

See [EXTENSION_DISCUSSION.md](EXTENSION_DISCUSSION.md) for details.

---

## 🔬 Research Code Ethos

### Mathematical Rigor

STOK-Core prioritizes **correctness over performance**:
- Exact equation implementations (not approximations)
- All invariants enforced and tested
- GPU-accelerated but readable

**When in doubt**: Match the paper exactly, even if suboptimal.

### Documentation

**All public APIs should explain**:
- What it computes (mathematical definition)
- Where it's from (paper equation/theorem)
- How to use it (example)
- What it returns (interpretation)

**Example**:
```rust
/// Cumulative Feasibility Function κ*(x)
///
/// Computes the probability of achieving the goal while avoiding constraints.
///
/// # Mathematical Definition
///
/// From Equation [7] (Ringstrom & Schrater 2025):
/// ```text
/// κ*(x) = max_a [f₁(x,a) + f₂(x,a) Σ P(x'|x,a) κ*(x')]
/// ```
///
/// # Example
///
/// ```rust
/// let mdp = TaskMDP::simple_chain(10, 20, &device);
/// let stok = solve_task_mdp(&mdp)?;
/// println!("Feasibility: {:.3}", stok.kappa[0]);
/// ```
```

---

## 🧪 Testing Philosophy

### What to Test

**Every mathematical claim**:
- If you implement an equation, test that it produces correct results
- If paper claims a property (normalization, associativity), test it
- If a theorem states a bound, test the bound holds

**Integration over isolation**:
- Test end-to-end workflows (Task → STOK → Composition → Plan)
- Not just individual functions

### Test Organization

- **Unit tests**: In the same file as the implementation (`#[cfg(test)] mod tests { ... }`)
- **Integration tests**: cross-module workflows under `tests/<topic>_tests.rs`. Recently added: `tests/stok_tests.rs` (PP-105 invariants), `tests/composition_tests.rs` (PP-203 Eq [18-19]), `tests/sublimation_tests.rs` (PP-403/404 Theorem 2.4).
- **Examples**: full application scenarios (must run successfully).
- **Theorem-style tests**: when implementing a theorem, add a test that names the equation/theorem it pins (see `CHK-*` entries in `docs/dev/PAPER_ACCEPTANCE_CHECKLIST.md` for what counts as "passing").

**Standard**: 250+ tests, 0 failures. New code must maintain this.

---

## 🎓 Paper Reference

All implementations should match:

**Primary source**: Ringstrom & Schrater (2025), arXiv:2506.09499
**Secondary source**: Ringstrom PhD Thesis (2023), University of Minnesota

**When implementing**:
1. Read the relevant paper section
2. Identify the equation/algorithm number
3. Implement exactly as specified
4. Add equation reference in comments
5. Test against known results or properties

---

## 💻 Development Setup

### Prerequisites

- **Rust**: 1.75+ (2021 edition)
- **GPU**: Vulkan/Metal/DX12 support for WGPU backend
- **Optional**: CUDA 11+ for NVIDIA GPU acceleration

### Build Commands

```bash
# Standard development build
cargo build

# Release build (with LTO)
cargo build --release

# Run tests (all must pass!)
cargo test

# Run tests with output
cargo test -- --nocapture

# Check code (fast, no codegen)
cargo check

# Lint
cargo clippy

# Format
cargo fmt

# Documentation
cargo doc --open
```

### GPU Backend Selection

```bash
# Default: WGPU (cross-platform)
cargo test

# CPU backend (for testing without GPU)
# (requires code changes to use cpu_device())

# CUDA backend
cargo test --features cuda
```

---

## 📊 Quality Standards

### Before Submitting PR

**Checklist**:
- [ ] All tests pass: `cargo test`
- [ ] No clippy warnings: `cargo clippy`
- [ ] Code formatted: `cargo fmt`
- [ ] New features have tests
- [ ] Public APIs have rustdoc
- [ ] Examples run successfully
- [ ] No breaking changes (or clearly documented)

### Code Review Criteria

**We look for**:
- Mathematical correctness
- Test coverage
- Documentation quality
- Adherence to project style
- No performance regressions

---

## 🎯 Roadmap

### Active scope
See [`docs/dev/PAPER_PDF_TASKBOARD.md`](docs/dev/PAPER_PDF_TASKBOARD.md). Milestones M0, M1, M2, M4 are complete; M3 (factorization), M5 (Goal Kernel / Plan Kernel), M6 (Algorithm 2), and M7 (theorem option-set builders + final report) are the open work.

### Out of scope
Anything not on the PP-XXX board. In particular:
- Empowerment / preference construction (paper Section 3.C is discussion only)
- Real-time runtime (STOK-RT) and telemetry architecture
- Sparse tensor support
- Visualization tooling

See [`docs/EXTENSION_DISCUSSION.md`](docs/EXTENSION_DISCUSSION.md) for the long-running list of post-paper extensions.

---

## ❓ Questions?

- **General questions**: Open an [issue](https://github.com/your-repo/stok-core/issues)
- **Theory questions**: See [paper](https://arxiv.org/abs/2506.09499) or email authors
- **Implementation questions**: Check [rustdoc](https://docs.rs/stok-core) or open issue

---

## 📜 Code of Conduct

**Be respectful, constructive, and collaborative.**

This is research code supporting a published paper. Discussions should be:
- Focused on mathematical correctness and implementation quality
- Respectful of the theoretical framework
- Constructive in suggestions for improvements

---

## 🙏 Acknowledgments

Thank you for helping improve STOK-Core!

Special thanks to:
- Thomas J. Ringstrom & Paul R. Schrater for the theoretical framework
- The [Burn](https://github.com/tracel-ai/burn) team for the excellent ML framework
- All contributors and users

---

**Ready to contribute? Start by running the examples and exploring the code!** 🚀
