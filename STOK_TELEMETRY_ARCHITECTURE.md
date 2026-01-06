# STOK-CORE Telemetry & Interactive Runtime Architecture

**Version:** 1.0  
**Status:** Design Phase  
**Author:** STOK-CORE Development Team  
**Date:** 2025-01-02  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Vision & Goals](#2-vision--goals)
3. [System Architecture](#3-system-architecture)
4. [Core Components](#4-core-components)
5. [Phase Integration Strategy](#5-phase-integration-strategy)
6. [Data Flow Architecture](#6-data-flow-architecture)
7. [API Contracts](#7-api-contracts)
8. [Configuration System](#8-configuration-system)
9. [Extensibility Strategy](#9-extensibility-strategy)
10. [Implementation Roadmap](#10-implementation-roadmap)
11. [Testing Strategy](#11-testing-strategy)
12. [Performance Considerations](#12-performance-considerations)
13. [Security & Safety](#13-security--safety)

---

## 1. Executive Summary

This document defines the architecture for STOK-CORE's **telemetry and interactive runtime system**. The system provides:

- **Passive Observation**: Real-time visualization of algorithm state during execution
- **Active Control**: Runtime parameter adjustment and algorithm steering
- **Multi-Phase Support**: Extensible design supporting Phases 2-4 with minimal refactoring
- **Production Ready**: Performance-critical paths optimized for GPU workloads

**Key Technologies:**
- `tracing` - Structured event instrumentation
- `ratatui` - Terminal UI framework
- `crossterm` - Cross-platform terminal control
- Burn tensors - GPU-resident data extraction

**Design Principles:**
1. **Minimal Coupling**: Telemetry is opt-in and removable
2. **Zero-Cost Abstraction**: Disabled telemetry compiles to no-ops
3. **Functional Core**: Pure functions over stateful objects
4. **Progressive Enhancement**: Basic → Advanced features via configuration

---

## 2. Vision & Goals

### 2.1 Long-Term Vision

**STOK-CORE Telemetry** evolves into a comprehensive **algorithm observatory and control center**:

```
Phase 2: Monitor feasibility iteration convergence
         ↓
Phase 3: Visualize STOK composition chains
         ↓
Phase 4: Interactively explore tree search, prune branches, adjust heuristics
         ↓
Future: Multi-agent coordination, distributed STOK training
```

**Interactive Capabilities (Roadmap):**
- **Pause/Resume**: Freeze algorithm mid-execution
- **Parameter Tuning**: Adjust convergence tolerance, search depth during runtime
- **Breakpoints**: Stop on specific conditions (e.g., κ < threshold)
- **State Inspection**: Drill down into individual tensors
- **Comparative Runs**: A/B test different configurations side-by-side

### 2.2 Phase 2 Goals (MVP)

**Scope**: Feasibility Iteration Only

**Must Have:**
- [x] Display iteration count and convergence delta
- [x] Show κ convergence curve (sparkline)
- [x] Detect and flag anomalies (policy degeneracy, normalization violations)
- [x] Emit structured logs for offline analysis

**Should Have:**
- [ ] Pause/resume iteration
- [ ] Adjust epsilon tolerance during runtime
- [ ] Export snapshots to file

**Won't Have (Phase 2):**
- Composition visualization
- Tree search interaction
- Multi-threaded coordination

### 2.3 Success Criteria

| Metric | Target |
|--------|--------|
| Instrumentation overhead (disabled) | 0% |
| Instrumentation overhead (INFO level) | < 1% |
| Instrumentation overhead (DEBUG level) | < 5% |
| Instrumentation overhead (TRACE level) | < 20% (acceptable for debugging) |
| TUI responsiveness | < 100ms input latency |
| Snapshot write time | < 50ms for 1000-state problem |

---

## 3. System Architecture

### 3.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      STOK-CORE                              │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ Solver     │  │ Composition  │  │ Planning     │        │
│  │ (Phase 2)  │  │ (Phase 3)    │  │ (Phase 4)    │        │
│  └─────┬──────┘  └──────┬───────┘  └──────┬───────┘        │
│        │                │                  │                │
│        └────────────────┴──────────────────┘                │
│                         │                                   │
│                    tracing::event!()                        │
│                         │                                   │
└─────────────────────────┼───────────────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────┐
         │   tracing_subscriber           │
         │   ┌──────────────────────┐     │
         │   │ EnvFilter            │     │  RUST_LOG controls
         │   └──────────────────────┘     │  what gets emitted
         │   ┌──────────────────────┐     │
         │   │ fmt::Layer (stdout)  │     │  For offline logs
         │   └──────────────────────┘     │
         │   ┌──────────────────────┐     │
         │   │ TelemetryLayer       │     │  Bridge to TUI
         │   │   (mpsc::Sender)     │     │
         │   └──────────┬───────────┘     │
         └──────────────┼─────────────────┘
                        │
                  mpsc::channel
                        │
                        ▼
         ┌──────────────────────────────┐
         │   Ratatui TUI                │
         │  ┌────────────────────────┐  │
         │  │ Event Loop (thread)    │  │
         │  │  - Receive telemetry   │  │
         │  │  - Handle user input   │  │
         │  │  - Render UI           │  │
         │  └────────────────────────┘  │
         │  ┌────────────────────────┐  │
         │  │ State Management       │  │
         │  │  - Iteration metrics   │  │
         │  │  - History buffers     │  │
         │  │  - UI state            │  │
         │  └────────────────────────┘  │
         │  ┌────────────────────────┐  │
         │  │ Control Commands       │  │
         │  │  → mpsc::Sender        │  │
         │  └────────┬───────────────┘  │
         └───────────┼──────────────────┘
                     │
              mpsc::channel
                     │
                     ▼
         ┌───────────────────────────────┐
         │   Runtime Controller          │
         │   (solver reads commands)     │
         └───────────────────────────────┘
```

### 3.2 Architectural Layers

| Layer | Responsibility | Technologies |
|-------|----------------|--------------|
| **Instrumentation** | Emit structured events from algorithms | `tracing` macros |
| **Collection** | Aggregate and route events | `tracing_subscriber::Layer` |
| **Extraction** | GPU→CPU scalar transfers | Burn tensor ops |
| **Transport** | Event delivery to consumers | `std::sync::mpsc` |
| **Visualization** | Render live dashboard | `ratatui`, `crossterm` |
| **Control** | Bidirectional runtime commands | `mpsc` channels |
| **Persistence** | Save snapshots and logs | File I/O |

### 3.3 Module Organization

```
src/
├── telemetry/
│   ├── mod.rs                  # Public API
│   ├── fields.rs               # Metric extraction (GPU→CPU)
│   ├── events.rs               # Event type definitions
│   ├── layer.rs                # Custom tracing::Layer impl
│   ├── snapshots.rs            # Full tensor dumps
│   ├── tui/
│   │   ├── mod.rs
│   │   ├── app.rs              # TUI application state
│   │   ├── ui.rs               # Rendering logic
│   │   ├── input.rs            # User input handling
│   │   ├── widgets/
│   │   │   ├── convergence.rs  # κ convergence widget
│   │   │   ├── metrics.rs      # Real-time metrics widget
│   │   │   └── tree.rs         # Tree search widget (Phase 4)
│   │   └── themes.rs           # Color schemes
│   └── control/
│       ├── mod.rs
│       ├── commands.rs         # Runtime control commands
│       └── controller.rs       # Command dispatcher
```

---

## 4. Core Components

### 4.1 Event Taxonomy

**Event Types** (from STOK-CORE algorithms):

```rust
/// Events emitted during execution
#[derive(Debug, Clone)]
pub enum TelemetryEvent {
    // ===== Phase 2: Feasibility Iteration =====
    FeasibilityIterationStart {
        n_states: usize,
        n_actions: usize,
        max_iterations: usize,
        tolerance: f32,
    },
    
    IterationComplete {
        iteration: usize,
        delta: f32,
        kappa_max: f32,
        kappa_min: f32,
        policy_changes: usize,
    },
    
    ConvergenceReached {
        iterations: usize,
        final_delta: f32,
        reason: ConvergenceReason,
    },
    
    StokConstructionStart {
        max_time: usize,
    },
    
    StokConstructionComplete {
        duration_ms: u64,
    },
    
    AnomalyDetected {
        kind: AnomalyKind,
        severity: Severity,
        details: String,
    },
    
    // ===== Phase 3: Composition =====
    CompositionStart {
        stok1_id: String,
        stok2_id: String,
        strategy: CompositionStrategy,
    },
    
    CompositionComplete {
        composed_id: String,
        duration_ms: u64,
        time_horizon: usize,
    },
    
    // ===== Phase 4: Planning =====
    TreeSearchStart {
        initial_state: usize,
        target_goal: Option<GoalId>,
        max_depth: usize,
    },
    
    NodeExpanded {
        node_id: usize,
        depth: usize,
        state: usize,
        feasibility: f32,
    },
    
    NodePruned {
        node_id: usize,
        reason: PruningReason,
    },
    
    PlanFound {
        plan_id: usize,
        depth: usize,
        feasibility: f32,
        expected_time: Option<f32>,
    },
    
    TreeSearchComplete {
        nodes_expanded: usize,
        nodes_pruned: usize,
        plans_found: usize,
        duration_ms: u64,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum AnomalyKind {
    PolicyDegeneracy,        // All states select same action
    NormalizationViolation,  // Row sums != 1.0
    KappaInconsistency,     // κ != sum(η⁺)
    LargeConvergenceDelta,  // Δκ > expected
    NumericalInstability,   // NaN or Inf detected
}

#[derive(Debug, Clone, Copy)]
pub enum Severity {
    Info,
    Warning,
    Error,
    Critical,
}
```

### 4.2 Metric Extraction (GPU→CPU Bridge)

**Design Principle**: Minimize transfers, batch operations.

```rust
// src/telemetry/fields.rs

/// Extract scalar metrics from GPU tensors
/// 
/// PERFORMANCE: Single batched transfer of ~10 scalars
pub fn extract_iteration_metrics<B: Backend>(
    kappa_prev: &Tensor<B, 1>,
    kappa_new: &Tensor<B, 1>,
    policy_prev: &Tensor<B, 1, Int>,
    policy_new: &Tensor<B, 1, Int>,
) -> IterationMetrics {
    // All reductions happen on GPU
    let delta = (kappa_new.clone() - kappa_prev.clone())
        .abs()
        .max();
    
    let kappa_max = kappa_new.clone().max();
    let kappa_min = kappa_new.clone().min();
    let kappa_mean = kappa_new.clone().mean();
    
    let policy_changes = policy_new.clone()
        .not_equal(policy_prev.clone())
        .int()
        .sum();
    
    // Batch transfer: ~5 scalars in one GPU→CPU op
    let scalars = Tensor::cat(
        vec![delta, kappa_max, kappa_min, kappa_mean, policy_changes.float()],
        0
    )
    .into_data()
    .to_vec()
    .unwrap();
    
    IterationMetrics {
        delta: scalars[0],
        kappa_max: scalars[1],
        kappa_min: scalars[2],
        kappa_mean: scalars[3],
        policy_changes: scalars[4] as usize,
    }
}

/// Extract STOK health metrics
pub fn extract_stok_health<B: Backend>(
    eta_plus: &Tensor<B, 3>,
    eta_minus: &Tensor<B, 3>,
    kappa: &Tensor<B, 1>,
) -> StokHealthMetrics {
    // row_mass = sum over (x_f, t_f) for each x_i
    let eta_total = eta_plus.clone() + eta_minus.clone();
    let row_mass = eta_total
        .sum_dim(2)  // Sum over time
        .sum_dim(1); // Sum over final states
    
    let row_mass_min = row_mass.clone().min();
    let row_mass_max = row_mass.clone().max();
    
    // κ-η consistency: kappa should equal sum(eta_plus)
    let kappa_from_eta = eta_plus.clone()
        .sum_dim(2)
        .sum_dim(1);
    
    let kappa_eta_diff = (kappa.clone() - kappa_from_eta)
        .abs()
        .max();
    
    let scalars = Tensor::cat(
        vec![row_mass_min, row_mass_max, kappa_eta_diff],
        0
    )
    .into_data()
    .to_vec()
    .unwrap();
    
    StokHealthMetrics {
        row_mass_min: scalars[0],
        row_mass_max: scalars[1],
        kappa_eta_consistency: scalars[2],
    }
}
```

**Performance Contract:**
- Extraction called at most once per iteration
- Maximum of 20 scalars transferred per extraction
- No full tensor copies unless explicitly triggered (TRACE level)

### 4.3 Tracing Layer Implementation

```rust
// src/telemetry/layer.rs

use tracing::{Event, Metadata, Subscriber};
use tracing_subscriber::layer::{Context, Layer};
use std::sync::mpsc::Sender;

/// Custom tracing layer that bridges to TUI
pub struct TelemetryLayer {
    tx: Sender<TelemetryEvent>,
}

impl TelemetryLayer {
    pub fn new(tx: Sender<TelemetryEvent>) -> Self {
        Self { tx }
    }
}

impl<S: Subscriber> Layer<S> for TelemetryLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        // Parse structured fields
        let mut visitor = TelemetryVisitor::default();
        event.record(&mut visitor);
        
        // Convert to TelemetryEvent
        if let Some(telem_event) = visitor.into_event(event.metadata()) {
            // Non-blocking send (drop if TUI slow)
            let _ = self.tx.try_send(telem_event);
        }
    }
}

#[derive(Default)]
struct TelemetryVisitor {
    iteration: Option<usize>,
    delta: Option<f32>,
    kappa_max: Option<f32>,
    kappa_min: Option<f32>,
    policy_changes: Option<usize>,
    // ... other fields
}

impl tracing::field::Visit for TelemetryVisitor {
    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        match field.name() {
            "iteration" => self.iteration = Some(value as usize),
            "policy_changes" => self.policy_changes = Some(value as usize),
            _ => {}
        }
    }
    
    fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
        match field.name() {
            "delta" => self.delta = Some(value as f32),
            "kappa_max" => self.kappa_max = Some(value as f32),
            "kappa_min" => self.kappa_min = Some(value as f32),
            _ => {}
        }
    }
    
    fn record_debug(&mut self, _field: &tracing::field::Field, _value: &dyn std::fmt::Debug) {}
}

impl TelemetryVisitor {
    fn into_event(self, meta: &Metadata) -> Option<TelemetryEvent> {
        // Match on span name + fields to construct appropriate event
        match meta.target() {
            "stok_core::solver::feasibility_iteration" => {
                if let (Some(iter), Some(delta), Some(kappa_max), Some(kappa_min), Some(changes)) = 
                    (self.iteration, self.delta, self.kappa_max, self.kappa_min, self.policy_changes) {
                    Some(TelemetryEvent::IterationComplete {
                        iteration: iter,
                        delta,
                        kappa_max,
                        kappa_min,
                        policy_changes: changes,
                    })
                } else {
                    None
                }
            },
            _ => None,
        }
    }
}
```

### 4.4 Ratatui TUI Application

```rust
// src/telemetry/tui/app.rs

use std::sync::mpsc::{Receiver, Sender};
use crossterm::event::{KeyCode, KeyEvent};

pub struct TelemetryApp {
    // Event reception
    event_rx: Receiver<TelemetryEvent>,
    
    // Command transmission
    cmd_tx: Sender<RuntimeCommand>,
    
    // Application state
    state: AppState,
    
    // UI configuration
    config: TuiConfig,
}

#[derive(Default)]
pub struct AppState {
    // Phase 2: Feasibility Iteration
    pub fi_state: FeasibilityIterationState,
    
    // Phase 3: Composition (future)
    pub composition_state: Option<CompositionState>,
    
    // Phase 4: Planning (future)
    pub planning_state: Option<PlanningState>,
    
    // Global
    pub paused: bool,
    pub current_phase: Phase,
}

#[derive(Default)]
pub struct FeasibilityIterationState {
    pub current_iteration: usize,
    pub max_iterations: usize,
    
    // Metric histories (ring buffers)
    pub delta_history: RingBuffer<f32>,      // Last 100 points
    pub kappa_max_history: RingBuffer<f32>,
    pub policy_changes_history: RingBuffer<usize>,
    
    // Current snapshot
    pub current_delta: f32,
    pub current_kappa_max: f32,
    pub current_kappa_min: f32,
    
    // Convergence status
    pub converged: bool,
    pub convergence_reason: Option<ConvergenceReason>,
}

impl TelemetryApp {
    pub fn new(
        event_rx: Receiver<TelemetryEvent>,
        cmd_tx: Sender<RuntimeCommand>,
        config: TuiConfig,
    ) -> Self {
        Self {
            event_rx,
            cmd_tx,
            state: AppState::default(),
            config,
        }
    }
    
    /// Main event loop
    pub fn run(&mut self, terminal: &mut Terminal<impl Backend>) -> io::Result<()> {
        loop {
            // Process incoming telemetry events
            while let Ok(event) = self.event_rx.try_recv() {
                self.handle_telemetry_event(event);
            }
            
            // Render UI
            terminal.draw(|f| self.render(f))?;
            
            // Handle user input
            if crossterm::event::poll(std::time::Duration::from_millis(100))? {
                if let crossterm::event::Event::Key(key) = crossterm::event::read()? {
                    if self.handle_key_event(key)? {
                        break; // Exit requested
                    }
                }
            }
            
            // Exit condition
            if self.should_exit() {
                break;
            }
        }
        
        Ok(())
    }
    
    fn handle_telemetry_event(&mut self, event: TelemetryEvent) {
        match event {
            TelemetryEvent::IterationComplete { 
                iteration, delta, kappa_max, kappa_min, policy_changes 
            } => {
                let fi = &mut self.state.fi_state;
                fi.current_iteration = iteration;
                fi.current_delta = delta;
                fi.current_kappa_max = kappa_max;
                fi.current_kappa_min = kappa_min;
                
                fi.delta_history.push(delta);
                fi.kappa_max_history.push(kappa_max);
                fi.policy_changes_history.push(policy_changes);
            },
            
            TelemetryEvent::ConvergenceReached { iterations, final_delta, reason } => {
                let fi = &mut self.state.fi_state;
                fi.converged = true;
                fi.convergence_reason = Some(reason);
            },
            
            // Handle other events...
            _ => {}
        }
    }
    
    fn handle_key_event(&mut self, key: KeyEvent) -> io::Result<bool> {
        match key.code {
            KeyCode::Char('q') => Ok(true), // Exit
            KeyCode::Char(' ') => {
                // Toggle pause
                self.state.paused = !self.state.paused;
                self.cmd_tx.send(RuntimeCommand::TogglePause)?;
                Ok(false)
            },
            KeyCode::Char('+') => {
                // Increase epsilon (looser convergence)
                self.cmd_tx.send(RuntimeCommand::AdjustEpsilon(1.5))?;
                Ok(false)
            },
            KeyCode::Char('-') => {
                // Decrease epsilon (tighter convergence)
                self.cmd_tx.send(RuntimeCommand::AdjustEpsilon(0.67))?;
                Ok(false)
            },
            KeyCode::Char('s') => {
                // Snapshot current state
                self.cmd_tx.send(RuntimeCommand::TakeSnapshot)?;
                Ok(false)
            },
            _ => Ok(false),
        }
    }
}
```

### 4.5 Runtime Control Commands

```rust
// src/telemetry/control/commands.rs

/// Commands sent from TUI to solver
#[derive(Debug, Clone)]
pub enum RuntimeCommand {
    // Execution control
    Pause,
    Resume,
    TogglePause,
    Step,                    // Single iteration
    
    // Parameter adjustment
    AdjustEpsilon(f32),      // Multiply current epsilon by factor
    SetMaxIterations(usize),
    
    // Inspection
    TakeSnapshot,
    DumpState { path: PathBuf },
    
    // Phase 4: Tree search control
    PruneNode { node_id: usize },
    SetSearchDepth(usize),
    
    // Termination
    Abort,
}

/// Controller that dispatches commands to algorithm
pub struct RuntimeController {
    cmd_rx: Receiver<RuntimeCommand>,
    state: Arc<Mutex<ControlState>>,
}

#[derive(Default)]
pub struct ControlState {
    pub paused: bool,
    pub epsilon_multiplier: f32,
    pub snapshot_requested: bool,
    pub abort_requested: bool,
}

impl RuntimeController {
    pub fn new(cmd_rx: Receiver<RuntimeCommand>) -> Self {
        Self {
            cmd_rx,
            state: Arc::new(Mutex::new(ControlState::default())),
        }
    }
    
    /// Check for pending commands (non-blocking)
    pub fn poll(&self) {
        while let Ok(cmd) = self.cmd_rx.try_recv() {
            self.handle_command(cmd);
        }
    }
    
    fn handle_command(&self, cmd: RuntimeCommand) {
        let mut state = self.state.lock().unwrap();
        match cmd {
            RuntimeCommand::Pause => state.paused = true,
            RuntimeCommand::Resume => state.paused = false,
            RuntimeCommand::TogglePause => state.paused = !state.paused,
            RuntimeCommand::AdjustEpsilon(factor) => {
                state.epsilon_multiplier *= factor;
            },
            RuntimeCommand::TakeSnapshot => state.snapshot_requested = true,
            RuntimeCommand::Abort => state.abort_requested = true,
            _ => {}
        }
    }
    
    pub fn is_paused(&self) -> bool {
        self.state.lock().unwrap().paused
    }
    
    pub fn get_epsilon_multiplier(&self) -> f32 {
        self.state.lock().unwrap().epsilon_multiplier
    }
    
    pub fn should_abort(&self) -> bool {
        self.state.lock().unwrap().abort_requested
    }
}
```

---

## 5. Phase Integration Strategy

### 5.1 Phase 2: Feasibility Iteration (Current)

**Instrumentation Points:**

```rust
// src/solver/feasibility_iteration.rs

pub fn feasibility_iteration<B: Backend>(
    mdp: &TaskMDP<B>,
    config: FeasibilityIterationConfig,
) -> Result<FeasibilityIterationResult<B>, StokError> {
    // 🔍 TOP-LEVEL SPAN
    let span = tracing::info_span!(
        "feasibility_iteration",
        n_states = mdp.n_states(),
        n_actions = mdp.n_actions(),
    );
    let _enter = span.enter();
    
    tracing::info!("Starting feasibility iteration");
    
    // Optional: Initialize runtime controller
    let controller = config.runtime_controller.as_ref();
    
    // Main loop
    for iteration in 0..config.convergence.max_iterations {
        // 🎛️ CONTROL: Check for pause/abort
        if let Some(ctrl) = controller {
            ctrl.poll();
            
            while ctrl.is_paused() {
                std::thread::sleep(Duration::from_millis(100));
                ctrl.poll();
            }
            
            if ctrl.should_abort() {
                tracing::warn!("Aborted by user");
                return Err(StokError::Aborted);
            }
        }
        
        // Bellman backup
        let (kappa_new, policy_new) = bellman_backup_kappa(&kappa, mdp);
        
        // 📊 TELEMETRY: Extract metrics (if DEBUG enabled)
        if tracing::enabled!(tracing::Level::DEBUG) {
            let metrics = extract_iteration_metrics(
                &kappa_prev, 
                &kappa_new,
                &policy_prev,
                &policy_new,
            );
            
            tracing::debug!(
                iteration,
                delta = %metrics.delta,
                kappa_max = %metrics.kappa_max,
                kappa_min = %metrics.kappa_min,
                policy_changes = metrics.policy_changes,
                "Iteration complete"
            );
            
            // 🚨 ANOMALY DETECTION
            if metrics.delta > 0.5 {
                tracing::warn!(
                    iteration,
                    delta = %metrics.delta,
                    "Large convergence delta detected"
                );
            }
        }
        
        kappa = kappa_new;
        policy = policy_new;
        
        // Convergence check...
    }
    
    // STOK construction with health check
    let kernel = construct_stok(mdp, &kappa, &policy, config.max_time)?;
    
    if tracing::enabled!(tracing::Level::DEBUG) {
        let health = extract_stok_health(&kernel.eta_plus, &kernel.eta_minus, &kernel.kappa);
        
        tracing::debug!(
            row_mass_min = %health.row_mass_min,
            row_mass_max = %health.row_mass_max,
            "STOK health check"
        );
        
        if health.row_mass_min < 0.99 {
            tracing::error!("STOK normalization violated");
        }
    }
    
    Ok(result)
}
```

**Configuration Extension:**

```rust
pub struct FeasibilityIterationConfig {
    // Existing fields...
    pub convergence: ConvergenceConfig,
    pub max_time: usize,
    
    // NEW: Telemetry support
    pub runtime_controller: Option<Arc<RuntimeController>>,
}
```

### 5.2 Phase 3: Composition (Future)

**Instrumentation Points:**

```rust
// src/composition/chapman_kolmogorov.rs

pub fn compose_stoks<B: Backend>(
    stok1: &STOKKernel<B>,
    stok2: &STOKKernel<B>,
) -> Result<ComposedSTOK<B>, StokError> {
    let span = tracing::info_span!(
        "compose_stoks",
        stok1_states = stok1.n_states(),
        stok2_states = stok2.n_states(),
        composed_time = stok1.max_time() + stok2.max_time() - 1,
    );
    let _enter = span.enter();
    
    tracing::info!("Starting STOK composition");
    
    // Composition loop with progress events
    for t_mu in 0..t_composed {
        if tracing::enabled!(tracing::Level::DEBUG) && t_mu % 10 == 0 {
            tracing::debug!(
                t_mu,
                progress = (t_mu as f32 / t_composed as f32) * 100.0,
                "Composition progress"
            );
        }
        
        // ... composition logic
    }
    
    tracing::info!(duration_ms = start.elapsed().as_millis(), "Composition complete");
    
    Ok(result)
}
```

**TUI Widget:**
- Progress bar for composition
- Visualization of intermediate time slices
- Memory usage tracking

### 5.3 Phase 4: Planning/Tree Search (Future)

**Instrumentation Points:**

```rust
// src/planning/tree_search.rs

pub fn tree_search<B: Backend>(
    goal_kernel: &GoalKernel<B>,
    initial_state: usize,
    config: TreeSearchConfig,
) -> SearchResult {
    let span = tracing::info_span!(
        "tree_search",
        initial_state,
        max_depth = config.max_depth,
    );
    let _enter = span.enter();
    
    while let Some(node) = queue.pop_front() {
        tracing::trace!(
            node_id = node.id,
            depth = node.depth,
            state = node.state,
            feasibility = %node.cumulative_feasibility,
            "Expanding node"
        );
        
        // Expansion logic...
        
        if should_prune(&node, &config) {
            tracing::debug!(
                node_id = node.id,
                reason = ?prune_reason,
                "Pruning node"
            );
            continue;
        }
    }
    
    tracing::info!(
        nodes_expanded = stats.nodes_expanded,
        nodes_pruned = stats.nodes_pruned,
        "Tree search complete"
    );
    
    result
}
```

**TUI Widget:**
- Tree visualization (ASCII art or interactive graph)
- Node inspection on hover/selection
- Real-time pruning decisions
- **Interactive control**: Click node to prune manually

---

## 6. Data Flow Architecture

### 6.1 Event Flow Diagram

```
GPU Computation
     │
     │ (Tensor operations)
     │
     ▼
extract_iteration_metrics()
     │
     │ (Single batched CPU transfer: ~10 scalars)
     │
     ▼
IterationMetrics struct
     │
     │
     ▼
tracing::debug!()
     │
     │ (Structured event)
     │
     ▼
tracing_subscriber::Layer
     │
     ├──► fmt::Layer ──► stdout/file
     │
     └──► TelemetryLayer
             │
             │ (Parse fields, construct TelemetryEvent)
             │
             ▼
         mpsc::Sender
             │
             │ (Non-blocking try_send)
             │
             ▼
         mpsc::Receiver
             │
             │ (Polled by TUI thread)
             │
             ▼
     TelemetryApp::handle_event()
             │
             │ (Update AppState)
             │
             ▼
     RingBuffer::push()
             │
             │
             ▼
     terminal.draw() ──► Ratatui render
```

### 6.2 Control Flow Diagram

```
User Input (TUI)
     │
     │ (Keyboard: Space, +, -, s, etc.)
     │
     ▼
TelemetryApp::handle_key_event()
     │
     │ (Construct RuntimeCommand)
     │
     ▼
mpsc::Sender<RuntimeCommand>
     │
     │ (Send to solver thread)
     │
     ▼
mpsc::Receiver<RuntimeCommand>
     │
     │
     ▼
RuntimeController::poll()
     │
     │ (Update ControlState)
     │
     ▼
Arc<Mutex<ControlState>>
     │
     │
     ▼
Solver checks:
  - ctrl.is_paused() → spin wait
  - ctrl.should_abort() → return early
  - ctrl.get_epsilon_multiplier() → adjust config
```

### 6.3 Thread Model

```
┌─────────────────────────────────────────────────┐
│             Main Thread                         │
│  ┌───────────────────────────────────────────┐  │
│  │  STOK Solver (GPU operations)             │  │
│  │  - Feasibility iteration                  │  │
│  │  - Composition                            │  │
│  │  - Tree search                            │  │
│  │                                           │  │
│  │  Polls: RuntimeController::poll()        │  │
│  │  Emits: tracing::debug!()                │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
                     ║
                     ║ mpsc::channel
                     ║ (TelemetryEvent)
                     ▼
┌─────────────────────────────────────────────────┐
│             TUI Thread                          │
│  ┌───────────────────────────────────────────┐  │
│  │  Event Loop (100ms poll)                  │  │
│  │  - Receive telemetry events               │  │
│  │  - Handle user input                      │  │
│  │  - Render UI                              │  │
│  │  - Send runtime commands                  │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
                     ║
                     ║ mpsc::channel
                     ║ (RuntimeCommand)
                     ▼
                Main Thread
            (RuntimeController)
```

**Threading Strategy:**
- Solver runs in main thread (GPU affinity)
- TUI runs in separate thread (responsive UI)
- Communication via `mpsc` (lock-free)
- TUI uses `try_send` (never blocks solver)
- Solver uses `try_recv` (checks commands periodically)

---

## 7. API Contracts

### 7.1 Public API

```rust
// src/telemetry/mod.rs

/// Initialize telemetry subscriber (call once at program start)
pub fn init_telemetry() -> Result<(), TelemetryError>;

/// Initialize telemetry with TUI
pub fn init_telemetry_with_tui(config: TuiConfig) -> Result<TelemetryHandle, TelemetryError>;

/// TUI configuration
#[derive(Clone, Debug)]
pub struct TuiConfig {
    pub theme: Theme,
    pub update_rate_ms: u64,
    pub history_size: usize,
}

impl Default for TuiConfig {
    fn default() -> Self {
        Self {
            theme: Theme::Dark,
            update_rate_ms: 100,
            history_size: 100,
        }
    }
}

/// Handle for controlling telemetry
pub struct TelemetryHandle {
    tui_thread: JoinHandle<()>,
    cmd_tx: Sender<RuntimeCommand>,
}

impl TelemetryHandle {
    /// Send command to solver
    pub fn send_command(&self, cmd: RuntimeCommand) -> Result<(), TelemetryError>;
    
    /// Shutdown TUI gracefully
    pub fn shutdown(self) -> Result<(), TelemetryError>;
}
```

### 7.2 Integration API (for Solver)

```rust
// What solvers need to do:

// 1. Add runtime controller to config
let (cmd_tx, cmd_rx) = mpsc::channel();
let controller = RuntimeController::new(cmd_rx);

let mut config = FeasibilityIterationConfig::default();
config.runtime_controller = Some(Arc::new(controller));

// 2. Run solver
let result = feasibility_iteration(&mdp, config)?;

// Done! Instrumentation is automatic via tracing macros.
```

### 7.3 Metric Extraction API

```rust
// src/telemetry/fields.rs

/// Extract iteration metrics (Phase 2)
pub fn extract_iteration_metrics<B: Backend>(
    kappa_prev: &Tensor<B, 1>,
    kappa_new: &Tensor<B, 1>,
    policy_prev: &Tensor<B, 1, Int>,
    policy_new: &Tensor<B, 1, Int>,
) -> IterationMetrics;

/// Extract STOK health metrics (Phase 2)
pub fn extract_stok_health<B: Backend>(
    eta_plus: &Tensor<B, 3>,
    eta_minus: &Tensor<B, 3>,
    kappa: &Tensor<B, 1>,
) -> StokHealthMetrics;

/// Extract composition metrics (Phase 3)
pub fn extract_composition_metrics<B: Backend>(
    composed: &ComposedSTOK<B>,
) -> CompositionMetrics;

/// Extract tree search metrics (Phase 4)
pub fn extract_search_node_metrics(
    node: &SearchNode,
) -> SearchNodeMetrics;
```

---

## 8. Configuration System

### 8.1 Environment-Based Configuration

```bash
# Logging level (tracing filter)
RUST_LOG=stok_core=debug

# Specific modules
RUST_LOG=stok_core::solver=trace,stok_core::composition=info

# Enable TUI
STOK_TUI=1

# TUI theme
STOK_TUI_THEME=light  # or dark

# Snapshot directory
STOK_SNAPSHOT_DIR=./snapshots

# Performance profiling
STOK_PROFILE=1  # Emit performance spans
```

### 8.2 Programmatic Configuration

```rust
use stok_core::telemetry::{TelemetryConfig, init_telemetry_with_config};

let config = TelemetryConfig {
    enable_tui: true,
    tui_config: TuiConfig {
        theme: Theme::Dark,
        update_rate_ms: 50,  // High refresh rate
        history_size: 200,
    },
    snapshot_dir: PathBuf::from("./debug_snapshots"),
    enable_profiling: true,
};

let handle = init_telemetry_with_config(config)?;
```

### 8.3 Feature Flags (Cargo.toml)

```toml
[features]
default = []
telemetry = ["tracing", "tracing-subscriber"]
tui = ["telemetry", "ratatui", "crossterm"]
profiling = ["telemetry", "tracing-flame"]

# For production builds
minimal = []
```

**Usage:**
```bash
# Development: full telemetry
cargo run --features tui

# Production: no telemetry overhead
cargo build --release --no-default-features --features minimal
```

---

## 9. Extensibility Strategy

### 9.1 Plugin Architecture (Future)

```rust
/// Trait for custom telemetry handlers
pub trait TelemetryPlugin: Send + Sync {
    fn on_event(&mut self, event: &TelemetryEvent);
    fn name(&self) -> &str;
}

/// Example: Prometheus exporter
pub struct PrometheusPlugin {
    registry: Registry,
}

impl TelemetryPlugin for PrometheusPlugin {
    fn on_event(&mut self, event: &TelemetryEvent) {
        match event {
            TelemetryEvent::IterationComplete { iteration, delta, .. } => {
                self.registry.record_gauge("stok_delta", *delta);
                self.registry.record_counter("stok_iterations", *iteration as f64);
            },
            _ => {}
        }
    }
    
    fn name(&self) -> &str { "prometheus" }
}
```

### 9.2 Custom Widget API

```rust
// src/telemetry/tui/widgets/mod.rs

pub trait TelemetryWidget {
    fn render(&self, frame: &mut Frame, area: Rect, state: &AppState);
    fn handle_input(&mut self, key: KeyEvent) -> bool;
}

// Example: Custom Phase 4 tree visualization
pub struct TreeWidget {
    selected_node: Option<usize>,
    zoom_level: f32,
}

impl TelemetryWidget for TreeWidget {
    fn render(&self, frame: &mut Frame, area: Rect, state: &AppState) {
        // Render tree using ratatui primitives
        // ...
    }
    
    fn handle_input(&mut self, key: KeyEvent) -> bool {
        match key.code {
            KeyCode::Up => { self.selected_node = self.selected_node.map(|n| n.saturating_sub(1)); true },
            KeyCode::Down => { self.selected_node = Some(self.selected_node.unwrap_or(0) + 1); true },
            _ => false,
        }
    }
}
```

### 9.3 Versioning Strategy

**Event Schema Versioning:**
```rust
#[derive(Debug, Clone)]
pub enum TelemetryEvent {
    V1(TelemetryEventV1),
    V2(TelemetryEventV2),  // Future
}

// Maintain backward compatibility
impl From<TelemetryEventV1> for TelemetryEvent {
    fn from(v1: TelemetryEventV1) -> Self {
        TelemetryEvent::V1(v1)
    }
}
```

---

## 10. Implementation Roadmap

### 10.1 Phase 2 MVP (Week 1-2)

**Deliverables:**
- [ ] `tracing` instrumentation in `feasibility_iteration.rs`
- [ ] `extract_iteration_metrics()` and `extract_stok_health()`
- [ ] `TelemetryLayer` implementation
- [ ] Basic Ratatui TUI (progress bar, convergence sparkline, metrics display)
- [ ] `RuntimeCommand::Pause` and `RuntimeCommand::Resume`
- [ ] Unit tests for metric extraction
- [ ] Integration test with simple_chain MDP

**Success Criteria:**
- TUI displays live convergence for 100-state problem
- Pause/resume works without race conditions
- Overhead < 5% at DEBUG level

### 10.2 Phase 2 Polish (Week 3)

**Deliverables:**
- [ ] Anomaly detection (normalization violations, policy degeneracy)
- [ ] Snapshot system (save full κ/π to disk)
- [ ] `RuntimeCommand::AdjustEpsilon`
- [ ] Themed UI (dark/light modes)
- [ ] Documentation and examples
- [ ] Performance benchmarks

### 10.3 Phase 3 Integration (Week 4-5)

**Deliverables:**
- [ ] Composition progress events
- [ ] Composition time horizon visualization
- [ ] Memory usage tracking
- [ ] Multi-composition sequence viewer

### 10.4 Phase 4 Integration (Week 6-8)

**Deliverables:**
- [ ] Tree search node events
- [ ] Interactive tree widget (ASCII art initially)
- [ ] Node pruning commands
- [ ] Plan comparison view
- [ ] Beam search parameter tuning

### 10.5 Future Enhancements

- [ ] Distributed tracing (multiple STOK solvers)
- [ ] Web UI (export TUI state to browser via WebSocket)
- [ ] Replay system (record/playback events)
- [ ] A/B testing dashboard (compare configurations)
- [ ] ML-based anomaly detection

---

## 11. Testing Strategy

### 11.1 Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_extract_iteration_metrics() {
        let device = default_device();
        let kappa_prev = Tensor::from_floats([0.0, 0.5, 1.0], &device);
        let kappa_new = Tensor::from_floats([0.1, 0.6, 1.0], &device);
        let policy_prev = Tensor::from_ints([0, 1, 2], &device);
        let policy_new = Tensor::from_ints([0, 2, 2], &device);
        
        let metrics = extract_iteration_metrics(
            &kappa_prev, &kappa_new,
            &policy_prev, &policy_new,
        );
        
        assert!((metrics.delta - 0.1).abs() < 1e-6);
        assert_eq!(metrics.policy_changes, 1);
    }
    
    #[test]
    fn test_telemetry_layer_parsing() {
        // Test that TelemetryVisitor correctly parses tracing events
        // ...
    }
}
```

### 11.2 Integration Tests

```rust
#[test]
fn test_feasibility_iteration_with_tui() {
    use std::sync::mpsc;
    
    let (event_tx, event_rx) = mpsc::channel();
    let (cmd_tx, cmd_rx) = mpsc::channel();
    
    // Initialize telemetry layer
    let telemetry_layer = TelemetryLayer::new(event_tx);
    
    // Initialize tracing
    tracing_subscriber::registry()
        .with(telemetry_layer)
        .init();
    
    // Spawn TUI thread (headless for testing)
    let tui_handle = std::thread::spawn(move || {
        let mut events = vec![];
        while let Ok(event) = event_rx.recv_timeout(Duration::from_secs(5)) {
            events.push(event);
            if matches!(event, TelemetryEvent::ConvergenceReached { .. }) {
                break;
            }
        }
        events
    });
    
    // Run solver
    let device = default_device();
    let mdp = TaskMDP::simple_chain(10, 20, &device);
    let controller = RuntimeController::new(cmd_rx);
    let mut config = FeasibilityIterationConfig::default();
    config.runtime_controller = Some(Arc::new(controller));
    
    let result = feasibility_iteration(&mdp, config).unwrap();
    
    // Verify telemetry events
    let events = tui_handle.join().unwrap();
    assert!(!events.is_empty());
    assert!(events.iter().any(|e| matches!(e, TelemetryEvent::ConvergenceReached { .. })));
}
```

### 11.3 Performance Tests

```rust
#[test]
fn test_telemetry_overhead() {
    let device = default_device();
    let mdp = TaskMDP::simple_chain(100, 50, &device);
    
    // Baseline: no telemetry
    let start = Instant::now();
    let _ = feasibility_iteration(&mdp, FeasibilityIterationConfig::default()).unwrap();
    let baseline_time = start.elapsed();
    
    // With DEBUG-level telemetry
    init_telemetry();
    std::env::set_var("RUST_LOG", "stok_core=debug");
    
    let start = Instant::now();
    let _ = feasibility_iteration(&mdp, FeasibilityIterationConfig::default()).unwrap();
    let telemetry_time = start.elapsed();
    
    let overhead = (telemetry_time.as_secs_f64() / baseline_time.as_secs_f64()) - 1.0;
    assert!(overhead < 0.05, "Overhead {}% exceeds 5%", overhead * 100.0);
}
```

---

## 12. Performance Considerations

### 12.1 Optimization Strategies

**Rule 1: Compile-time elimination**
- `tracing` macros are zero-cost when disabled
- Use feature flags to strip telemetry entirely in production

**Rule 2: Lazy evaluation**
- Only extract metrics if `tracing::enabled!(Level::DEBUG)`
- Use non-blocking `try_send` to avoid blocking solver

**Rule 3: Batched transfers**
- Combine multiple scalar extractions into single GPU→CPU transfer
- Maximum 20 scalars per iteration

**Rule 4: Sampling**
- Don't log every iteration at TRACE level
- Sample: log every Nth iteration based on problem size

**Rule 5: Ring buffers**
- Fixed-size history buffers (e.g., last 100 points)
- Avoid unbounded memory growth

### 12.2 Memory Budget

| Component | Memory (100-state problem) |
|-----------|----------------------------|
| History buffers (100 points × 5 metrics) | ~2 KB |
| AppState | ~10 KB |
| Event queue (bounded at 1000) | ~100 KB |
| Total TUI overhead | < 150 KB |

**No impact on GPU memory** - telemetry only extracts scalars.

### 12.3 Profiling Hooks

```rust
// src/telemetry/profiling.rs

#[cfg(feature = "profiling")]
pub fn enable_profiling() {
    use tracing_flame::FlameLayer;
    
    let (flame_layer, _guard) = FlameLayer::with_file("./stok_trace.folded").unwrap();
    
    tracing_subscriber::registry()
        .with(flame_layer)
        .init();
}
```

**Usage:**
```bash
cargo build --release --features profiling
./target/release/stok-example
inferno-flamegraph stok_trace.folded > flamegraph.svg
```

---

## 13. Security & Safety

### 13.1 Thread Safety

**Invariants:**
1. `mpsc` channels provide lock-free communication
2. `Arc<Mutex<ControlState>>` protects shared state
3. TUI thread never holds GPU locks
4. Solver thread never blocks on TUI

**Race Condition Prevention:**
- Commands processed via atomic `try_recv` (no blocking)
- State updates protected by mutex
- No shared mutable state between threads

### 13.2 Resource Limits

**Bounded Channels:**
```rust
// Use bounded channel to prevent memory explosion
let (tx, rx) = mpsc::sync_channel(1000); // Max 1000 pending events
```

**Graceful Degradation:**
- If TUI slow, drop events (solver never blocks)
- If disk full, log error but continue execution
- If memory constrained, reduce history buffer size

### 13.3 Error Handling

```rust
pub enum TelemetryError {
    ChannelClosed,
    TuiInitFailed(String),
    IoError(io::Error),
    InvalidConfig(String),
}

impl From<io::Error> for TelemetryError {
    fn from(e: io::Error) -> Self {
        TelemetryError::IoError(e)
    }
}
```

**Failure Modes:**
- Telemetry failure **never** crashes solver
- Solver can run without telemetry (degraded experience)
- TUI panic is caught and logged

---

## Appendix A: Example Usage

### A.1 Basic Usage (Logging Only)

```rust
use stok_core::prelude::*;
use stok_core::telemetry::init_telemetry;

fn main() {
    // Initialize with RUST_LOG=info
    init_telemetry().unwrap();
    
    let device = default_device();
    let mdp = TaskMDP::simple_chain(100, 50, &device);
    
    // Logs go to stdout
    let result = feasibility_iteration(&mdp, Default::default()).unwrap();
    
    println!("Converged in {} iterations", result.iterations());
}
```

**Output:**
```
INFO stok_core::solver: Starting feasibility iteration n_states=100
INFO stok_core::solver: Converged iterations=42 final_delta=0.000098
Converged in 42 iterations
```

### A.2 Interactive TUI

```rust
use stok_core::prelude::*;
use stok_core::telemetry::{init_telemetry_with_tui, TuiConfig};

fn main() {
    // Spawn TUI
    let handle = init_telemetry_with_tui(TuiConfig::default()).unwrap();
    
    let device = default_device();
    let mdp = TaskMDP::simple_chain(100, 50, &device);
    
    // Solver runs, TUI displays live progress
    let result = feasibility_iteration(&mdp, Default::default()).unwrap();
    
    // Shutdown TUI
    handle.shutdown().unwrap();
}
```

**TUI Controls:**
- `Space` - Pause/Resume
- `+/-` - Adjust epsilon
- `s` - Take snapshot
- `q` - Quit

### A.3 Programmatic Control

```rust
use stok_core::prelude::*;
use stok_core::telemetry::{RuntimeController, RuntimeCommand};
use std::sync::mpsc;

fn main() {
    let (cmd_tx, cmd_rx) = mpsc::channel();
    let controller = Arc::new(RuntimeController::new(cmd_rx));
    
    // Spawn control thread
    let cmd_tx_clone = cmd_tx.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(2));
        cmd_tx_clone.send(RuntimeCommand::Pause).unwrap();
        
        println!("Paused! Adjust epsilon? (y/n)");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        
        if input.trim() == "y" {
            cmd_tx_clone.send(RuntimeCommand::AdjustEpsilon(0.5)).unwrap();
        }
        
        cmd_tx_clone.send(RuntimeCommand::Resume).unwrap();
    });
    
    // Run solver
    let device = default_device();
    let mdp = TaskMDP::simple_chain(100, 50, &device);
    let mut config = FeasibilityIterationConfig::default();
    config.runtime_controller = Some(controller);
    
    let result = feasibility_iteration(&mdp, config).unwrap();
}
```

---

## Appendix B: Troubleshooting

### B.1 Common Issues

**Issue**: TUI not displaying events  
**Cause**: `RUST_LOG` not set or too restrictive  
**Fix**: `export RUST_LOG=stok_core=debug`

**Issue**: High overhead (>10%)  
**Cause**: TRACE level enabled  
**Fix**: Use DEBUG for normal operation, TRACE only for debugging

**Issue**: TUI freezes  
**Cause**: Event queue full (solver producing events faster than TUI consumes)  
**Fix**: Increase channel capacity or reduce update rate

**Issue**: Snapshots not saving  
**Cause**: Directory doesn't exist  
**Fix**: Create `./snapshots` or set `STOK_SNAPSHOT_DIR`

### B.2 Debugging

**Enable tracing diagnostics:**
```rust
use tracing_subscriber::fmt;

fmt()
    .with_max_level(tracing::Level::TRACE)
    .with_thread_ids(true)
    .with_line_number(true)
    .init();
```

**Inspect event stream:**
```bash
RUST_LOG=trace cargo run 2>&1 | grep "stok_core::solver"
```

---

## Appendix C: Future Research Directions

1. **Distributed Telemetry**: Aggregate events from multiple STOK solvers running on different GPUs
2. **ML-Powered Anomaly Detection**: Train classifier to detect convergence issues
3. **Automatic Hyperparameter Tuning**: Use telemetry to guide epsilon/depth search
4. **Causal Profiling**: Identify which algorithm components are bottlenecks
5. **Predictive Dashboards**: Forecast convergence time based on early iterations

---

**End of Document**

*This architecture is designed to evolve with STOK-CORE. Feedback and refinements welcome as implementation progresses.*
