# Contributing to STOK-Core

Thank you for your interest in contributing to STOK-Core, the reference implementation of Option Kernel Bellman Equations!

---

## 🎯 Project Status

**STOK-Core is a complete reference implementation** (100/100 paper parity) of:
> Ringstrom, T., & Schrater, P. (2025). *A Unified Theory of Compositionality, Modularity, and Interpretability in Markov Decision Processes.* arXiv:2506.09499

All core algorithms, theorems, and examples are implemented and validated.

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

- **Unit tests**: In same file as implementation (`mod tests { ... }`)
- **Integration tests**: Cross-module workflows
- **Examples**: Full application scenarios (must run successfully!)

**Standard**: 220+ tests, 0 failures. New code must maintain this.

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

### Current (1.0.0 - Complete!)
- ✅ All 4 theorems
- ✅ All core equations
- ✅ 4 working examples
- ✅ 220+ tests

### Potential Future Work
- Product-space tree search
- Sparse tensor support
- Empowerment implementation
- Visualization tools
- Real-time runtime (STOK-RT)

See [EXTENSION_DISCUSSION.md](EXTENSION_DISCUSSION.md) for detailed analysis.

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
