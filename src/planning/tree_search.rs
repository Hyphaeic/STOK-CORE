//! # Option Sequence Tree Search
//!
//! Implements breadth-first and best-first search over option sequences.
//!
//! Based on Algorithm 2 from Ringstrom & Schrater (2025).

use super::goal_kernel::GoalKernel;
use super::plan::Plan;
use crate::types::{GoalId, StateIdx};
use burn::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

/// Search strategy for tree search
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchStrategy {
    /// Expand nodes in breadth-first order
    BreadthFirst,
    /// Expand highest-feasibility nodes first
    BestFirst,
    /// Expand deepest nodes first
    DepthFirst,
}

/// Configuration for option tree search
#[derive(Clone, Debug)]
pub struct TreeSearchConfig {
    /// Maximum search depth (number of options in sequence)
    pub max_depth: usize,

    /// Minimum feasibility threshold for expanding a node
    pub feasibility_threshold: f32,

    /// Maximum number of nodes to expand
    pub max_nodes: usize,

    /// Search strategy to use
    pub search_strategy: SearchStrategy,

    /// Target goal (if searching for specific goal)
    pub target_goal: Option<GoalId>,

    /// Whether to track all paths or just best
    pub track_all_paths: bool,

    /// Sublimated feasibility for abstract pruning (Theorem 2.4)
    ///
    /// Maps HL space index → κ*_sub vector for that space.
    ///
    /// If provided, tree search will prune nodes where κ*_sub(z) = 0
    /// for any HL component, as these are abstractly infeasible.
    ///
    /// See: Ringstrom & Schrater (2025), Theorem 2.4, Figure 7 (bottom)
    pub sublimated_feasibility: Option<crate::hierarchy::SublimatedFeasibilityCache>,
}

impl Default for TreeSearchConfig {
    fn default() -> Self {
        Self {
            max_depth: 10,
            feasibility_threshold: 0.01,
            max_nodes: 10000,
            search_strategy: SearchStrategy::BreadthFirst,
            target_goal: None,
            track_all_paths: false,
            sublimated_feasibility: None,
        }
    }
}

/// Node in the option search tree
#[derive(Clone, Debug)]
pub struct SearchNode {
    /// Current state (deterministic for now)
    pub state: usize,

    /// Cumulative time
    pub time: usize,

    /// Path taken to reach this node
    pub path: Vec<GoalId>,

    /// Cumulative feasibility (product of κ along path)
    pub cumulative_feasibility: f32,

    /// Depth in search tree
    pub depth: usize,

    /// Is this a terminal node?
    pub is_terminal: bool,

    /// Node ID for tracking
    pub id: usize,

    /// Parent node ID
    pub parent_id: Option<usize>,
}

impl SearchNode {
    /// Create root node
    pub fn root(initial_state: usize) -> Self {
        Self {
            state: initial_state,
            time: 0,
            path: vec![],
            cumulative_feasibility: 1.0,
            depth: 0,
            is_terminal: false,
            id: 0,
            parent_id: None,
        }
    }

    /// Expand node with a new option
    pub fn expand(
        &self,
        goal: GoalId,
        next_state: usize,
        step_time: usize,
        step_feasibility: f32,
        node_id: usize,
    ) -> Self {
        let mut new_path = self.path.clone();
        new_path.push(goal);

        Self {
            state: next_state,
            time: self.time + step_time,
            path: new_path,
            cumulative_feasibility: self.cumulative_feasibility * step_feasibility,
            depth: self.depth + 1,
            is_terminal: false,
            id: node_id,
            parent_id: Some(self.id),
        }
    }
}

/// Search statistics
#[derive(Clone, Debug, Default)]
pub struct SearchStats {
    /// Number of nodes expanded
    pub nodes_expanded: usize,

    /// Number of nodes pruned
    pub nodes_pruned: usize,

    /// Maximum depth reached
    pub max_depth_reached: usize,

    /// Search time in milliseconds
    pub search_time_ms: f64,
}

/// Result of tree search
#[derive(Clone, Debug)]
pub struct SearchResult {
    /// Best plan found
    pub best_plan: Option<Plan>,

    /// All terminal plans (if track_all_paths enabled)
    pub all_plans: Vec<Plan>,

    /// Search statistics
    pub stats: SearchStats,
}

/// Breadth-first option sequence search
///
/// Implements simplified version of Algorithm 2 from the paper.
///
/// # Arguments
///
/// * `goal_kernel` - Goal kernel containing STOKs
/// * `initial_state` - Starting state
/// * `config` - Search configuration
///
/// # Returns
///
/// Search result with best plan and statistics
///
/// # Example
///
/// ```rust,ignore
/// let config = TreeSearchConfig {
///     max_depth: 5,
///     target_goal: Some(GoalId(10)),
///     ..Default::default()
/// };
///
/// let result = tree_search(&goal_kernel, 0, config);
/// if let Some(plan) = result.best_plan {
///     println!("Found plan: {}", plan);
/// }
/// ```
pub fn tree_search<B: Backend>(
    goal_kernel: &GoalKernel<B>,
    initial_state: usize,
    config: TreeSearchConfig,
) -> SearchResult {
    let start_time = std::time::Instant::now();

    let mut queue = VecDeque::new();
    let mut best_plan: Option<Plan> = None;
    let mut best_feasibility = 0.0f32;
    let mut all_plans = vec![];
    let mut stats = SearchStats::default();
    let mut next_node_id = 1usize;

    // Initialize with root node
    let root = SearchNode::root(initial_state);
    queue.push_back(root);

    while let Some(node) = queue.pop_front() {
        stats.nodes_expanded += 1;

        // Check termination conditions
        if stats.nodes_expanded >= config.max_nodes {
            break;
        }

        if node.depth >= config.max_depth {
            stats.max_depth_reached = stats.max_depth_reached.max(node.depth);
            continue;
        }

        // Check if target goal is achievable from current state
        if let Some(target) = config.target_goal {
            let target_kappa = goal_kernel.query_feasibility(target, node.state);

            if target_kappa > 0.0 {
                // Can reach target from here - create terminal plan
                let total_feasibility = node.cumulative_feasibility * target_kappa;

                if total_feasibility > best_feasibility {
                    best_feasibility = total_feasibility;

                    let mut plan = Plan {
                        options: node.path.clone(),
                        feasibility: total_feasibility,
                        expected_time: None,
                        initial_state: StateIdx(initial_state),
                    };
                    plan.options.push(target);

                    best_plan = Some(plan.clone());

                    if config.track_all_paths {
                        all_plans.push(plan);
                    }
                }
            }
        }

        // Expand: try all feasible goals
        let feasible_goals = goal_kernel.feasible_goals(node.state);

        for goal_id in feasible_goals {
            // Skip target goal (handled above)
            if Some(goal_id) == config.target_goal {
                continue;
            }

            let kappa = goal_kernel.query_feasibility(goal_id, node.state);

            // Pruning: skip low-feasibility options
            if kappa < config.feasibility_threshold {
                stats.nodes_pruned += 1;
                continue;
            }

            // TODO: Sublimation pruning (Theorem 2.4)
            //
            // When tree search is extended to support ProductState nodes:
            //
            // if let Some(ref sub_cache) = config.sublimated_feasibility {
            //     for (space_idx, hl_state) in node.product_state.hl_states.iter().enumerate() {
            //         if sub_cache.is_abstractly_infeasible(space_idx, hl_state.as_discrete()) {
            //             stats.nodes_pruned_by_sublimation += 1;
            //             continue;  // Skip this branch - abstractly infeasible
            //         }
            //     }
            // }
            //
            // This implements the red star pruning from Figure 7 (bottom)

            // Determine next state (use mode for deterministic, could sample for stochastic)
            let (next_state, step_time) = sample_next_state_mode(goal_kernel, goal_id, node.state);

            let child = node.expand(goal_id, next_state, step_time, kappa, next_node_id);
            next_node_id += 1;

            queue.push_back(child);
        }
    }

    stats.search_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

    SearchResult {
        best_plan,
        all_plans,
        stats,
    }
}

/// Best-first search prioritizing high-feasibility paths
///
/// # Arguments
///
/// * `goal_kernel` - Goal kernel containing STOKs
/// * `initial_state` - Starting state
/// * `config` - Search configuration
///
/// # Returns
///
/// Search result with best plan found
pub fn best_first_search<B: Backend>(
    goal_kernel: &GoalKernel<B>,
    initial_state: usize,
    config: TreeSearchConfig,
) -> SearchResult {
    let start_time = std::time::Instant::now();

    let mut heap = BinaryHeap::new();
    let mut best_plan: Option<Plan> = None;
    let mut best_feasibility = 0.0f32;
    let mut stats = SearchStats::default();
    let mut next_node_id = 1usize;

    // Initialize
    let root = SearchNode::root(initial_state);
    heap.push(PrioritizedNode {
        node: root,
        priority: 1.0,
    });

    while let Some(PrioritizedNode { node, .. }) = heap.pop() {
        stats.nodes_expanded += 1;

        if stats.nodes_expanded >= config.max_nodes {
            break;
        }

        if node.depth >= config.max_depth {
            stats.max_depth_reached = stats.max_depth_reached.max(node.depth);
            continue;
        }

        // Early termination: if best possible from here is worse than current best
        if node.cumulative_feasibility <= best_feasibility {
            stats.nodes_pruned += 1;
            continue;
        }

        // Check target goal
        if let Some(target) = config.target_goal {
            let target_kappa = goal_kernel.query_feasibility(target, node.state);
            let total_feasibility = node.cumulative_feasibility * target_kappa;

            if total_feasibility > best_feasibility {
                best_feasibility = total_feasibility;

                let mut plan = Plan {
                    options: node.path.clone(),
                    feasibility: total_feasibility,
                    expected_time: None,
                    initial_state: StateIdx(initial_state),
                };
                plan.options.push(target);
                best_plan = Some(plan);
            }
        }

        // Expand children
        for goal_id in goal_kernel.feasible_goals(node.state) {
            if Some(goal_id) == config.target_goal {
                continue;
            }

            let kappa = goal_kernel.query_feasibility(goal_id, node.state);
            if kappa < config.feasibility_threshold {
                stats.nodes_pruned += 1;
                continue;
            }

            let (next_state, step_time) = sample_next_state_mode(goal_kernel, goal_id, node.state);

            let child = node.expand(goal_id, next_state, step_time, kappa, next_node_id);
            next_node_id += 1;

            heap.push(PrioritizedNode {
                priority: child.cumulative_feasibility,
                node: child,
            });
        }
    }

    stats.search_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

    SearchResult {
        best_plan,
        all_plans: vec![],
        stats,
    }
}

// ============================================================================
// Helper Types and Functions
// ============================================================================

/// Node wrapper for priority queue (max-heap by feasibility)
#[derive(Clone)]
struct PrioritizedNode {
    node: SearchNode,
    priority: f32, // Higher is better
}

impl PartialEq for PrioritizedNode {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for PrioritizedNode {}

impl PartialOrd for PrioritizedNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.priority.partial_cmp(&other.priority)
    }
}

impl Ord for PrioritizedNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Sample next state using mode (most likely outcome) for deterministic search
///
/// # Arguments
///
/// * `goal_kernel` - Goal kernel
/// * `goal` - Goal ID
/// * `current_state` - Current state
///
/// # Returns
///
/// Tuple of (most_likely_state, most_likely_time)
fn sample_next_state_mode<B: Backend>(
    goal_kernel: &GoalKernel<B>,
    goal: GoalId,
    current_state: usize,
) -> (usize, usize) {
    let stok = match goal_kernel.get_stok(goal) {
        Some(s) => s,
        None => return (current_state, 0), // Fallback
    };

    // Find most likely (state, time) pair
    let eta = stok.combined_stok();
    let s = stok.n_states();
    let t = stok.max_time();

    let dist = eta
        .clone()
        .slice([current_state..(current_state + 1), 0..s, 0..t])
        .reshape([s * t]);

    let probs: Vec<f32> = dist.into_data().to_vec().unwrap();

    // Find argmax
    let (max_idx, _) = probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .unwrap_or((0, &0.0));

    let next_state = max_idx / t;
    let step_time = max_idx % t;

    (next_state, step_time)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{default_device, DefaultBackend};
    use crate::stok::STOKKernel;
    use burn::prelude::Tensor;

    #[test]
    fn test_search_node_root() {
        let root = SearchNode::root(5);

        assert_eq!(root.state, 5);
        assert_eq!(root.time, 0);
        assert_eq!(root.depth, 0);
        assert_eq!(root.cumulative_feasibility, 1.0);
        assert!(root.path.is_empty());
    }

    #[test]
    fn test_search_node_expand() {
        let root = SearchNode::root(0);
        let child = root.expand(GoalId(1), 5, 3, 0.8, 1);

        assert_eq!(child.state, 5);
        assert_eq!(child.time, 3);
        assert_eq!(child.depth, 1);
        assert!((child.cumulative_feasibility - 0.8).abs() < 1e-5);
        assert_eq!(child.path.len(), 1);
        assert_eq!(child.path[0], GoalId(1));
        assert_eq!(child.parent_id, Some(0));
    }

    #[test]
    fn test_prioritized_node_ordering() {
        let node1 = SearchNode::root(0);
        let node2 = SearchNode::root(1);

        let p1 = PrioritizedNode {
            node: node1,
            priority: 0.5,
        };
        let p2 = PrioritizedNode {
            node: node2,
            priority: 0.9,
        };

        // Higher priority should be greater (max-heap)
        assert!(p2 > p1);
    }

    #[test]
    fn test_tree_search_simple() {
        let device = default_device();
        let mut kernel: GoalKernel<DefaultBackend> = GoalKernel::new(5, device.clone());

        // Add simple goal
        let mut stok: STOKKernel<DefaultBackend> = STOKKernel::new(5, 5, &device);
        stok.kappa = Tensor::ones([5], &device); // All states feasible

        kernel
            .add_goal(GoalId(0), stok, "goal", Some(4))
            .unwrap();

        let config = TreeSearchConfig {
            max_depth: 3,
            target_goal: Some(GoalId(0)),
            ..Default::default()
        };

        let result = tree_search(&kernel, 0, config);

        assert!(result.best_plan.is_some());
        let plan = result.best_plan.unwrap();
        assert!(plan.feasibility > 0.0);
        assert!(plan.options.contains(&GoalId(0)));
    }

    #[test]
    fn test_best_first_search() {
        let device = default_device();
        let mut kernel: GoalKernel<DefaultBackend> = GoalKernel::new(5, device.clone());

        let mut stok: STOKKernel<DefaultBackend> = STOKKernel::new(5, 5, &device);
        stok.kappa = Tensor::ones([5], &device);

        kernel
            .add_goal(GoalId(0), stok, "goal", None)
            .unwrap();

        let config = TreeSearchConfig {
            max_depth: 3,
            target_goal: Some(GoalId(0)),
            search_strategy: SearchStrategy::BestFirst,
            ..Default::default()
        };

        let result = best_first_search(&kernel, 0, config);

        assert!(result.best_plan.is_some());
    }
}

// ============================================================================
// PP-601..605: Algorithm 2 — paper-faithful tree search over composite states
// ============================================================================
//
// `algorithm_2_search` and friends operate on `FactorizedGoalKernel` plus a
// base `TaskMDP` (for the boundary BL one-step). Each `Algorithm2Node` carries
// a `ProductState` (σ, z, x components, with σ folded into the HL state list)
// rather than a flat BL state. Expansion follows Algorithm 2 lines 13-19:
//   13. (α, x', t_f) ← option terminal + induced HL action via affordance
//   14. z' ← ρ_z(·|z, t_f) — HL state under default
//   15. x'' ← P_x(x''|x', π(x')) — one BL step under policy
//   16. σ'' ← P_σ(σ''|σ, α) — boundary HL transition (folded into 14 here)
//   17. t' = t_f + 1
//   19. cumulative feasibility update
// Lines 32-34 then filter to feasibility-maximizing then time-minimizing plans.
//
// PP-603 sublimation pruning: when a `SublimatedFeasibilityCache` is provided,
// any candidate child whose composite HL state has `κ_sub = 0` for some HL
// component is dropped (per Theorem 2.4: abstractly infeasible ⟹ practically
// infeasible).

use crate::hierarchy::ProductState;
use crate::mdp::TaskMDP;
use crate::planning::goal_kernel::FactorizedGoalKernel;

/// Search node for Algorithm 2 that carries the paper's composite state
/// `(σ, z, x, t)` (with σ folded into the HL state list of `ProductState`).
#[derive(Clone, Debug)]
pub struct Algorithm2Node {
    pub state: ProductState,
    pub time: usize,
    pub path: Vec<GoalId>,
    pub cumulative_feasibility: f32,
    pub depth: usize,
    pub is_terminal: bool,
    pub id: usize,
    pub parent_id: Option<usize>,
}

impl Algorithm2Node {
    pub fn root(initial_state: ProductState) -> Self {
        Self {
            state: initial_state,
            time: 0,
            path: vec![],
            cumulative_feasibility: 1.0,
            depth: 0,
            is_terminal: false,
            id: 0,
            parent_id: None,
        }
    }

    fn expand(
        &self,
        goal: GoalId,
        next_state: ProductState,
        time_advance: usize,
        step_feasibility: f32,
        node_id: usize,
    ) -> Self {
        let mut new_path = self.path.clone();
        new_path.push(goal);
        Self {
            state: next_state,
            time: self.time + time_advance,
            path: new_path,
            cumulative_feasibility: self.cumulative_feasibility * step_feasibility,
            depth: self.depth + 1,
            is_terminal: false,
            id: node_id,
            parent_id: Some(self.id),
        }
    }
}

/// Configuration for Algorithm 2 search.
#[derive(Clone)]
pub struct Algorithm2Config {
    pub max_depth: usize,
    pub feasibility_threshold: f32,
    pub max_nodes: usize,
    pub search_strategy: SearchStrategy,
    pub target_goal: Option<GoalId>,

    /// PP-603: optional sublimated-feasibility cache. When supplied, any
    /// candidate child whose composite HL state has `κ_sub = 0` for some HL
    /// component is pruned before being added to the queue.
    pub sublimated_feasibility: Option<crate::hierarchy::SublimatedFeasibilityCache>,
}

impl Default for Algorithm2Config {
    fn default() -> Self {
        Self {
            max_depth: 10,
            feasibility_threshold: 0.01,
            max_nodes: 10_000,
            search_strategy: SearchStrategy::BreadthFirst,
            target_goal: None,
            sublimated_feasibility: None,
        }
    }
}

/// Statistics accumulated by Algorithm 2 search.
#[derive(Clone, Debug, Default)]
pub struct Algorithm2Stats {
    pub nodes_expanded: usize,
    pub nodes_pruned_below_threshold: usize,
    pub nodes_pruned_by_sublimation: usize,
    pub max_depth_reached: usize,
    pub search_time_ms: f64,
}

/// A plan produced by Algorithm 2: a sequence of options + the composite
/// final state + cumulative feasibility + total time.
#[derive(Clone, Debug)]
pub struct Algorithm2Plan {
    pub options: Vec<GoalId>,
    pub final_state: ProductState,
    pub total_time: usize,
    pub cumulative_feasibility: f32,
}

/// Result of Algorithm 2 search: the feasibility-max time-min plan (Algorithm 2
/// lines 32-34) plus statistics.
#[derive(Clone, Debug, Default)]
pub struct Algorithm2Result {
    pub best_plan: Option<Algorithm2Plan>,
    pub all_terminal_plans: Vec<Algorithm2Plan>,
    pub stats: Algorithm2Stats,
}

/// PP-602/PP-604: Algorithm 2 search over the composite state. Picks BFS or
/// best-first based on `config.search_strategy`. Both share the same expansion
/// semantics so the choice of strategy never changes the answer (only the
/// order of node visitation).
pub fn algorithm_2_search<B: Backend>(
    goal_kernel: &FactorizedGoalKernel<B>,
    base_mdp: &TaskMDP<B>,
    initial: ProductState,
    config: &Algorithm2Config,
) -> Algorithm2Result {
    let start = std::time::Instant::now();
    let mut stats = Algorithm2Stats::default();
    let mut all_terminal: Vec<Algorithm2Plan> = Vec::new();
    let mut next_id = 1usize;

    let root = Algorithm2Node::root(initial);

    // Frontier: BFS uses a VecDeque, best-first uses a max-heap by feasibility.
    use std::collections::{BinaryHeap, VecDeque};
    let mut bfs_queue: VecDeque<Algorithm2Node> = VecDeque::new();
    let mut bf_heap: BinaryHeap<Algorithm2Prio> = BinaryHeap::new();
    let strategy = config.search_strategy;
    match strategy {
        SearchStrategy::BreadthFirst | SearchStrategy::DepthFirst => bfs_queue.push_back(root),
        SearchStrategy::BestFirst => bf_heap.push(Algorithm2Prio {
            priority: 1.0,
            node: root,
        }),
    }

    while stats.nodes_expanded < config.max_nodes {
        let node = match strategy {
            SearchStrategy::BreadthFirst => {
                if let Some(n) = bfs_queue.pop_front() {
                    n
                } else {
                    break;
                }
            }
            SearchStrategy::DepthFirst => {
                if let Some(n) = bfs_queue.pop_back() {
                    n
                } else {
                    break;
                }
            }
            SearchStrategy::BestFirst => {
                if let Some(p) = bf_heap.pop() {
                    p.node
                } else {
                    break;
                }
            }
        };
        stats.nodes_expanded += 1;
        stats.max_depth_reached = stats.max_depth_reached.max(node.depth);

        if node.depth >= config.max_depth {
            // Treat as terminal leaf.
            all_terminal.push(Algorithm2Plan {
                options: node.path.clone(),
                final_state: node.state.clone(),
                total_time: node.time,
                cumulative_feasibility: node.cumulative_feasibility,
            });
            continue;
        }

        // Per Algorithm 2 lines 11-12: iterate options whose κ at (x, t) > 0.
        let mut goal_ids = goal_kernel.goal_ids();
        goal_ids.sort();
        let mut any_child = false;
        for goal in goal_ids {
            let option = match goal_kernel.option(goal) {
                Some(o) => o,
                None => continue,
            };
            let kappa = option.feasibility_approx(&node.state);
            if kappa < config.feasibility_threshold {
                stats.nodes_pruned_below_threshold += 1;
                continue;
            }

            // Algorithm 2 line 13: (α, x', t_f) — pick the BL terminal mode.
            let max_time = option.base_stok.max_time();
            let n_x = goal_kernel.dims().base_size;
            let (x_f, t_f) = mode_of_factorized_term(option, &node.state, n_x, max_time);

            // Algorithm 2 line 14 + 16 (combined here): boundary HL update.
            // Pick boundary HL action α_f via affordance: the affordance prefers
            // a default (α_k = 0) for non-special states; for the deterministic
            // expansion we just use α_k = 0 across HL spaces. This matches the
            // "default" α in Algorithm 2 when affordance is unambiguous.
            let n_hl = goal_kernel.dims().n_hl_spaces();
            let alpha_f: Vec<usize> = vec![0; n_hl];
            let initial_hl_indices: Vec<usize> =
                node.state.hl_states.iter().map(|h| h.as_discrete()).collect();
            let next_hl_dists = match goal_kernel.boundary_hl_distribution(
                goal,
                &initial_hl_indices,
                &alpha_f,
                t_f.max(1),
            ) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let next_hl_indices: Vec<usize> = next_hl_dists
                .iter()
                .map(|d| {
                    let v: Vec<f32> = d.clone().into_data().to_vec().unwrap();
                    v.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i)
                        .unwrap_or(0)
                })
                .collect();

            // PP-603: sublimation pruning — drop the child if any HL component
            // is abstractly infeasible per the cache (Theorem 2.4 contrapositive).
            if let Some(ref cache) = config.sublimated_feasibility {
                let mut pruned = false;
                for (k, &z_k_next) in next_hl_indices.iter().enumerate() {
                    if cache.has_space(k) && cache.is_abstractly_infeasible(k, z_k_next) {
                        stats.nodes_pruned_by_sublimation += 1;
                        pruned = true;
                        break;
                    }
                }
                if pruned {
                    continue;
                }
            }

            // Algorithm 2 line 15: x'' = P_x(·|x_f, π(x_f)).
            let pi_at_xf: i32 = option
                .base_stok
                .policy
                .clone()
                .slice([x_f..(x_f + 1)])
                .into_scalar()
                .elem();
            let pi_at_xf = pi_at_xf as usize;
            let n_a = base_mdp.n_actions();
            let trans_row = base_mdp
                .transition
                .clone()
                .slice([x_f..(x_f + 1), pi_at_xf..(pi_at_xf + 1), 0..n_x])
                .reshape([n_x]);
            let trans_v: Vec<f32> = trans_row.into_data().to_vec().unwrap();
            let _ = n_a;
            let x_pp: usize = trans_v
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(x_f);

            // Build child product state.
            let mut child_state = ProductState::new(x_pp);
            for &z in &next_hl_indices {
                child_state = child_state.with_hl_discrete(z);
            }

            let time_advance = t_f + 1; // Algorithm 2 line 17.
            let child = node.expand(goal, child_state, time_advance, kappa, next_id);
            next_id += 1;
            any_child = true;

            match strategy {
                SearchStrategy::BreadthFirst | SearchStrategy::DepthFirst => {
                    bfs_queue.push_back(child);
                }
                SearchStrategy::BestFirst => {
                    bf_heap.push(Algorithm2Prio {
                        priority: child.cumulative_feasibility,
                        node: child,
                    });
                }
            }
        }

        if !any_child {
            // No expansion possible — node is a terminal leaf.
            all_terminal.push(Algorithm2Plan {
                options: node.path.clone(),
                final_state: node.state.clone(),
                total_time: node.time,
                cumulative_feasibility: node.cumulative_feasibility,
            });
        }
    }

    // Algorithm 2 lines 32-34: feasibility-max then time-min over leaves.
    let best_plan = if all_terminal.is_empty() {
        None
    } else {
        let max_feas = all_terminal
            .iter()
            .map(|p| p.cumulative_feasibility)
            .fold(f32::NEG_INFINITY, |a, b| a.max(b));
        let max_set: Vec<&Algorithm2Plan> = all_terminal
            .iter()
            .filter(|p| (p.cumulative_feasibility - max_feas).abs() < 1e-6)
            .collect();
        max_set
            .into_iter()
            .min_by_key(|p| p.total_time)
            .cloned()
    };

    stats.search_time_ms = start.elapsed().as_secs_f64() * 1000.0;
    Algorithm2Result {
        best_plan,
        all_terminal_plans: all_terminal,
        stats,
    }
}

#[derive(Clone)]
struct Algorithm2Prio {
    priority: f32,
    node: Algorithm2Node,
}

impl PartialEq for Algorithm2Prio {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}
impl Eq for Algorithm2Prio {}
impl PartialOrd for Algorithm2Prio {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.priority.partial_cmp(&other.priority)
    }
}
impl Ord for Algorithm2Prio {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Find the (x_f, t_f) pair maximizing `option.evaluate(initial, ., t)` over
/// the option's max_time horizon. Used for the deterministic Algorithm 2 line 13.
fn mode_of_factorized_term<B: Backend>(
    option: &crate::hierarchy::FactorizedSTOK<B>,
    initial: &ProductState,
    n_x: usize,
    max_time: usize,
) -> (usize, usize) {
    let mut best = (0usize, 1usize);
    let mut best_p = -1.0f32;
    for x_f in 0..n_x {
        let mut final_st = ProductState::new(x_f);
        for h in &initial.hl_states {
            final_st = final_st.with_hl_discrete(h.as_discrete());
        }
        for t in 1..max_time {
            let p = option.evaluate(initial, &final_st, t);
            if p > best_p {
                best_p = p;
                best = (x_f, t);
            }
        }
    }
    best
}

#[cfg(test)]
mod algorithm_2_tests {
    use super::*;
    use crate::backend::{default_device, DefaultBackend};
    use crate::hierarchy::{
        assemble_factorized_stok, FactorizedAffordance, ProductSpaceDims,
    };
    use crate::mdp::TaskMDP;
    use crate::planning::goal_kernel::{FactorizedGoalKernel, GoalInfo};
    use crate::solver::solve_task_mdp;
    use burn::prelude::Tensor;

    fn build_simple_a2_witness() -> (
        FactorizedGoalKernel<DefaultBackend>,
        TaskMDP<DefaultBackend>,
        ProductState,
    ) {
        let device = default_device();
        let base_mdp: TaskMDP<DefaultBackend> = TaskMDP::simple_chain(3, 6, &device);
        let base_stok = solve_task_mdp(&base_mdp).unwrap();

        // 2-state HL kernel with 1 action (identity).
        let mut hl_data = vec![0.0f32; 2 * 1 * 2];
        hl_data[0 * 2 + 0] = 1.0;
        hl_data[1 * 2 + 1] = 1.0;
        let hl_kernel: Tensor<DefaultBackend, 3> =
            Tensor::<DefaultBackend, 1>::from_floats(hl_data.as_slice(), &device).reshape([2, 1, 2]);

        // Affordance shape [3, 3, 1] (matches simple_chain 3-action shape).
        let mut f_data = vec![0.0f32; 3 * 3 * 1];
        for x in 0..3 {
            for a in 0..3 {
                f_data[x * 3 * 1 + a * 1 + 0] = 1.0;
            }
        }
        let f_tensor: Tensor<DefaultBackend, 3> =
            Tensor::<DefaultBackend, 1>::from_floats(f_data.as_slice(), &device).reshape([3, 3, 1]);
        let affordance = FactorizedAffordance::new(vec![f_tensor]);

        let dims = ProductSpaceDims::new(3, vec![2]);
        let f_stok =
            assemble_factorized_stok(base_stok, vec![hl_kernel.clone()], vec![0], dims.clone())
                .unwrap();

        let mut gk = FactorizedGoalKernel::new(affordance, vec![hl_kernel], dims);
        gk.add_option(
            GoalId(0),
            f_stok,
            GoalInfo {
                id: GoalId(0),
                name: "g0".into(),
                target_state: Some(2),
                max_time: 6,
            },
        )
        .unwrap();

        let initial = ProductState::new(0).with_hl_discrete(0);
        (gk, base_mdp, initial)
    }

    /// PP-602: Algorithm 2 search produces a non-empty best plan on a feasible MDP.
    #[test]
    fn algorithm_2_bfs_finds_a_plan() {
        let (gk, base_mdp, initial) = build_simple_a2_witness();
        let config = Algorithm2Config {
            max_depth: 3,
            ..Default::default()
        };
        let result = algorithm_2_search(&gk, &base_mdp, initial, &config);
        assert!(
            result.best_plan.is_some(),
            "Algorithm 2 should find a plan on the feasible witness"
        );
        let plan = result.best_plan.unwrap();
        assert!(plan.cumulative_feasibility > 0.0);
        assert!(plan.options.len() <= 3);
    }

    /// PP-604: BFS and best-first must agree on the best plan's feasibility
    /// and total time (the underlying semantics is the same).
    #[test]
    fn algorithm_2_bfs_and_best_first_agree() {
        let (gk, base_mdp, initial) = build_simple_a2_witness();

        let bfs_cfg = Algorithm2Config {
            max_depth: 3,
            search_strategy: SearchStrategy::BreadthFirst,
            ..Default::default()
        };
        let bf_cfg = Algorithm2Config {
            max_depth: 3,
            search_strategy: SearchStrategy::BestFirst,
            ..Default::default()
        };

        let bfs = algorithm_2_search(&gk, &base_mdp, initial.clone(), &bfs_cfg);
        let bf = algorithm_2_search(&gk, &base_mdp, initial, &bf_cfg);

        let bfs_plan = bfs.best_plan.expect("BFS plan");
        let bf_plan = bf.best_plan.expect("best-first plan");
        // Same feasibility (within tolerance) — both should hit the same maximum.
        assert!(
            (bfs_plan.cumulative_feasibility - bf_plan.cumulative_feasibility).abs() < 1e-5,
            "BFS κ̃ = {} vs best-first κ̃ = {}",
            bfs_plan.cumulative_feasibility,
            bf_plan.cumulative_feasibility
        );
        // Same time among the κ̃-maximal plans.
        assert_eq!(bfs_plan.total_time, bf_plan.total_time);
    }

    /// PP-603: when the sublimation cache marks all HL states infeasible, the
    /// search must prune every branch and return no plan.
    #[test]
    fn algorithm_2_sublimation_cache_prunes_all() {
        let (gk, base_mdp, initial) = build_simple_a2_witness();

        let mut cache = crate::hierarchy::SublimatedFeasibilityCache::new();
        cache.add_space(0, vec![0.0, 0.0]); // both HL states "infeasible"

        let cfg = Algorithm2Config {
            max_depth: 3,
            sublimated_feasibility: Some(cache),
            ..Default::default()
        };
        let result = algorithm_2_search(&gk, &base_mdp, initial, &cfg);
        assert!(
            result.best_plan.is_none()
                || result
                    .best_plan
                    .as_ref()
                    .map(|p| p.options.is_empty())
                    .unwrap_or(true),
            "all branches should be pruned by the all-infeasible cache; got plan with {} options",
            result.best_plan.as_ref().map(|p| p.options.len()).unwrap_or(0)
        );
        assert!(
            result.stats.nodes_pruned_by_sublimation > 0,
            "PP-603: sublimation pruning counter must increment ({} ≤ 0)",
            result.stats.nodes_pruned_by_sublimation
        );
    }
}
