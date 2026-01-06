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
