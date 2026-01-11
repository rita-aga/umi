//! SimNetwork - Simulated Network with Fault Injection
//!
//! TigerStyle: Configurable network conditions with explicit fault injection.
//! Supports partitions, delays, packet loss, and message reordering.

use bytes::Bytes;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;

use super::clock::SimClock;
use super::fault::{FaultInjector, FaultType};
use super::rng::DeterministicRng;
use crate::constants::{
    NETWORK_JITTER_MS_DEFAULT, NETWORK_LATENCY_MS_DEFAULT, NETWORK_LATENCY_MS_MAX,
};

/// A network message in flight.
#[derive(Debug, Clone)]
pub struct NetworkMessage {
    /// Source node ID
    pub from: String,
    /// Destination node ID
    pub to: String,
    /// Message payload
    pub payload: Bytes,
    /// Time when message should be delivered (ms)
    pub deliver_at_ms: u64,
}

/// Network errors.
#[derive(Debug, Clone, thiserror::Error)]
pub enum NetworkError {
    /// Message was dropped due to partition
    #[error("network partition between {from} and {to}")]
    Partitioned {
        /// Source node that tried to send
        from: String,
        /// Destination node that couldn't be reached
        to: String,
    },

    /// Message was dropped due to fault injection
    #[error("packet loss fault injected")]
    PacketLoss,

    /// Connection timed out
    #[error("connection timeout")]
    Timeout,

    /// Connection refused
    #[error("connection refused")]
    ConnectionRefused,
}

/// Simulated network for DST.
///
/// TigerStyle:
/// - Deterministic message delivery with configurable delays
/// - Explicit partitions with heal/partition API
/// - Fault injection at send/receive boundaries
/// - Full statistics tracking
pub struct SimNetwork {
    /// Pending messages per destination node
    messages: Arc<RwLock<HashMap<String, VecDeque<NetworkMessage>>>>,
    /// Network partitions (set of (from, to) pairs that are partitioned)
    partitions: Arc<RwLock<Vec<(String, String)>>>,
    /// Simulation clock
    clock: SimClock,
    /// Fault injector (shared)
    fault_injector: Arc<FaultInjector>,
    /// RNG for latency jitter (RefCell for interior mutability)
    rng: RefCell<DeterministicRng>,
    /// Base latency in milliseconds
    base_latency_ms: u64,
    /// Latency jitter in milliseconds
    latency_jitter_ms: u64,
}

impl SimNetwork {
    /// Create a new simulated network.
    ///
    /// TigerStyle: Takes shared fault injector for consistent fault injection.
    #[must_use]
    pub fn new(clock: SimClock, rng: DeterministicRng, fault_injector: Arc<FaultInjector>) -> Self {
        Self {
            messages: Arc::new(RwLock::new(HashMap::new())),
            partitions: Arc::new(RwLock::new(Vec::new())),
            clock,
            fault_injector,
            rng: RefCell::new(rng),
            base_latency_ms: NETWORK_LATENCY_MS_DEFAULT,
            latency_jitter_ms: NETWORK_JITTER_MS_DEFAULT,
        }
    }

    /// Set network latency parameters.
    ///
    /// # Panics
    /// Panics if base_ms exceeds NETWORK_LATENCY_MS_MAX.
    #[must_use]
    pub fn with_latency(mut self, base_ms: u64, jitter_ms: u64) -> Self {
        // Precondition
        assert!(
            base_ms <= NETWORK_LATENCY_MS_MAX,
            "base_latency_ms {} exceeds max {}",
            base_ms,
            NETWORK_LATENCY_MS_MAX
        );

        self.base_latency_ms = base_ms;
        self.latency_jitter_ms = jitter_ms;
        self
    }

    /// Send a message from one node to another.
    ///
    /// Returns true if message was queued, false if dropped (partition/fault).
    pub async fn send(&self, from: &str, to: &str, payload: Bytes) -> bool {
        // Preconditions
        assert!(!from.is_empty(), "from node ID cannot be empty");
        assert!(!to.is_empty(), "to node ID cannot be empty");

        // Check for network partition
        {
            let partitions = self.partitions.read().await;
            if partitions
                .iter()
                .any(|(a, b)| (a == from && b == to) || (a == to && b == from))
            {
                tracing::debug!(from = from, to = to, "Message dropped: network partition");
                return false;
            }
        }

        // Check for packet loss fault
        if let Some(fault) = self.fault_injector.should_inject("network_send") {
            match fault {
                FaultType::NetworkTimeout
                | FaultType::NetworkConnectionRefused
                | FaultType::NetworkReset => {
                    tracing::debug!(from = from, to = to, fault = ?fault, "Message dropped: fault");
                    return false;
                }
                _ => {}
            }
        }

        // Calculate delivery time with latency
        let latency = self.calculate_latency();
        let deliver_at_ms = self.clock.now_ms() + latency;

        let message = NetworkMessage {
            from: from.to_string(),
            to: to.to_string(),
            payload,
            deliver_at_ms,
        };

        // Queue the message
        let mut messages = self.messages.write().await;
        messages
            .entry(to.to_string())
            .or_default()
            .push_back(message);

        true
    }

    /// Receive messages for a node.
    ///
    /// Returns all messages that have arrived (delivery time <= current time).
    pub async fn receive(&self, node_id: &str) -> Vec<NetworkMessage> {
        // Precondition
        assert!(!node_id.is_empty(), "node_id cannot be empty");

        let current_time = self.clock.now_ms();
        let mut messages = self.messages.write().await;

        let queue = match messages.get_mut(node_id) {
            Some(q) => q,
            None => return Vec::new(),
        };

        // Collect messages ready for delivery
        let mut ready = Vec::new();
        let mut remaining = VecDeque::new();

        while let Some(msg) = queue.pop_front() {
            if msg.deliver_at_ms <= current_time {
                ready.push(msg);
            } else {
                remaining.push_back(msg);
            }
        }

        *queue = remaining;

        // Check for message reordering fault
        if !ready.is_empty() {
            if let Some(FaultType::NetworkPartialWrite) =
                self.fault_injector.should_inject("network_receive")
            {
                self.rng.borrow_mut().shuffle(&mut ready);
                tracing::debug!(node_id = node_id, "Messages reordered by fault");
            }
        }

        ready
    }

    /// Create a network partition between two nodes.
    ///
    /// Messages between these nodes will be dropped.
    pub async fn partition(&self, node_a: &str, node_b: &str) {
        // Preconditions
        assert!(!node_a.is_empty(), "node_a cannot be empty");
        assert!(!node_b.is_empty(), "node_b cannot be empty");
        assert_ne!(node_a, node_b, "cannot partition node with itself");

        let mut partitions = self.partitions.write().await;
        partitions.push((node_a.to_string(), node_b.to_string()));

        tracing::info!(
            node_a = node_a,
            node_b = node_b,
            "Network partition created"
        );
    }

    /// Heal a network partition between two nodes.
    pub async fn heal(&self, node_a: &str, node_b: &str) {
        let mut partitions = self.partitions.write().await;
        partitions.retain(|(a, b)| !((a == node_a && b == node_b) || (a == node_b && b == node_a)));

        tracing::info!(node_a = node_a, node_b = node_b, "Network partition healed");
    }

    /// Heal all network partitions.
    pub async fn heal_all(&self) {
        let mut partitions = self.partitions.write().await;
        partitions.clear();

        tracing::info!("All network partitions healed");
    }

    /// Check if two nodes are partitioned.
    pub async fn is_partitioned(&self, node_a: &str, node_b: &str) -> bool {
        let partitions = self.partitions.read().await;
        partitions
            .iter()
            .any(|(a, b)| (a == node_a && b == node_b) || (a == node_b && b == node_a))
    }

    /// Get count of pending messages for a node.
    pub async fn pending_count(&self, node_id: &str) -> usize {
        let messages = self.messages.read().await;
        messages.get(node_id).map(|q| q.len()).unwrap_or(0)
    }

    /// Get total pending messages across all nodes.
    pub async fn total_pending(&self) -> usize {
        let messages = self.messages.read().await;
        messages.values().map(|q| q.len()).sum()
    }

    /// Clear all pending messages.
    pub async fn clear(&self) {
        let mut messages = self.messages.write().await;
        messages.clear();
    }

    /// Get the clock.
    #[must_use]
    pub fn clock(&self) -> &SimClock {
        &self.clock
    }

    /// Calculate latency with jitter.
    fn calculate_latency(&self) -> u64 {
        let jitter = if self.latency_jitter_ms > 0 {
            self.rng
                .borrow_mut()
                .next_usize(0, self.latency_jitter_ms as usize) as u64
        } else {
            0
        };
        self.base_latency_ms + jitter
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dst::fault::FaultInjectorBuilder;

    fn create_network() -> SimNetwork {
        let clock = SimClock::new();
        let mut rng = DeterministicRng::new(42);
        let fault_injector = Arc::new(FaultInjectorBuilder::new(rng.fork()).build());
        SimNetwork::new(clock, rng, fault_injector).with_latency(0, 0)
    }

    #[tokio::test]
    async fn test_send_and_receive() {
        let network = create_network();

        // Send message
        let sent = network.send("node-1", "node-2", Bytes::from("hello")).await;
        assert!(sent);

        // Receive message
        let messages = network.receive("node-2").await;
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].payload, Bytes::from("hello"));
        assert_eq!(messages[0].from, "node-1");
        assert_eq!(messages[0].to, "node-2");
    }

    #[tokio::test]
    async fn test_partition() {
        let network = create_network();

        // Create partition
        network.partition("node-1", "node-2").await;
        assert!(network.is_partitioned("node-1", "node-2").await);
        assert!(network.is_partitioned("node-2", "node-1").await); // Symmetric

        // Message should be dropped
        let sent = network.send("node-1", "node-2", Bytes::from("hello")).await;
        assert!(!sent);

        // Heal partition
        network.heal("node-1", "node-2").await;
        assert!(!network.is_partitioned("node-1", "node-2").await);

        // Message should go through
        let sent = network.send("node-1", "node-2", Bytes::from("hello")).await;
        assert!(sent);
    }

    #[tokio::test]
    async fn test_latency() {
        let clock = SimClock::new();
        let mut rng = DeterministicRng::new(42);
        let fault_injector = Arc::new(FaultInjectorBuilder::new(rng.fork()).build());
        let network = SimNetwork::new(clock.clone(), rng, fault_injector).with_latency(100, 0);

        // Send message
        network.send("node-1", "node-2", Bytes::from("hello")).await;

        // Should not be delivered yet
        let messages = network.receive("node-2").await;
        assert!(messages.is_empty());

        // Advance time
        clock.advance_ms(100);

        // Now should be delivered
        let messages = network.receive("node-2").await;
        assert_eq!(messages.len(), 1);
    }

    #[tokio::test]
    async fn test_multiple_messages() {
        let network = create_network();

        // Send multiple messages
        network.send("node-1", "node-2", Bytes::from("msg1")).await;
        network.send("node-1", "node-2", Bytes::from("msg2")).await;
        network.send("node-3", "node-2", Bytes::from("msg3")).await;

        assert_eq!(network.pending_count("node-2").await, 3);
        assert_eq!(network.total_pending().await, 3);

        // Receive all
        let messages = network.receive("node-2").await;
        assert_eq!(messages.len(), 3);
        assert_eq!(network.pending_count("node-2").await, 0);
    }

    #[tokio::test]
    async fn test_heal_all() {
        let network = create_network();

        // Create multiple partitions
        network.partition("node-1", "node-2").await;
        network.partition("node-2", "node-3").await;
        network.partition("node-1", "node-3").await;

        assert!(network.is_partitioned("node-1", "node-2").await);
        assert!(network.is_partitioned("node-2", "node-3").await);

        // Heal all
        network.heal_all().await;

        assert!(!network.is_partitioned("node-1", "node-2").await);
        assert!(!network.is_partitioned("node-2", "node-3").await);
        assert!(!network.is_partitioned("node-1", "node-3").await);
    }

    #[tokio::test]
    async fn test_clear() {
        let network = create_network();

        network.send("node-1", "node-2", Bytes::from("msg1")).await;
        network.send("node-1", "node-2", Bytes::from("msg2")).await;

        assert_eq!(network.total_pending().await, 2);

        network.clear().await;

        assert_eq!(network.total_pending().await, 0);
    }

    #[test]
    #[should_panic(expected = "from node ID cannot be empty")]
    fn test_send_empty_from() {
        let network = create_network();
        let _ = tokio_test::block_on(network.send("", "node-2", Bytes::from("hello")));
    }

    #[test]
    #[should_panic(expected = "cannot partition node with itself")]
    fn test_partition_self() {
        let network = create_network();
        let _ = tokio_test::block_on(network.partition("node-1", "node-1"));
    }
}
