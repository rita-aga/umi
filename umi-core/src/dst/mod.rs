//! DST - Deterministic Simulation Testing
//!
//! TigerBeetle/FoundationDB-style deterministic simulation testing framework.
//!
//! # Philosophy
//!
//! > "If you're not testing with fault injection, you're not testing."
//!
//! # Usage
//!
//! ```rust
//! use umi_core::dst::{Simulation, SimConfig, FaultConfig, FaultType};
//!
//! #[tokio::test]
//! async fn test_storage_survives_faults() {
//!     let sim = Simulation::new(SimConfig::with_seed(42))
//!         .with_fault(FaultConfig::new(FaultType::StorageWriteFail, 0.1));
//!
//!     sim.run(|env| async move {
//!         env.storage.write("key", b"value").await?;
//!         env.clock.advance_ms(1000);
//!         let result = env.storage.read("key").await?;
//!         assert_eq!(result, Some(b"value".to_vec()));
//!         Ok(())
//!     }).await.unwrap();
//! }
//! ```
//!
//! Run with explicit seed for reproducibility:
//! ```bash
//! DST_SEED=12345 cargo test
//! ```

mod clock;
mod config;
mod fault;
mod network;
mod property;
mod rng;
mod simulation;
mod storage;

pub use clock::SimClock;
pub use config::SimConfig;
pub use fault::{FaultConfig, FaultInjector, FaultInjectorBuilder, FaultType};
pub use network::{NetworkError, NetworkMessage, SimNetwork};
pub use property::{
    run_property_tests, test_seeds, PropertyTest, PropertyTestFailure, PropertyTestResult,
    PropertyTestable, TimeAdvanceConfig,
};
pub use rng::DeterministicRng;
pub use simulation::{create_simulation, SimEnvironment, Simulation};
pub use storage::{SimStorage, StorageError, StorageReadError, StorageWriteError};
