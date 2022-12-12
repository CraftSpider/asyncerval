//! Advanced async interval support
//!
//! # Features
//!
//! - Run setup, teardown, and error handling code easily
//! - Skip any number of ticks easily
//!
//! # Examples
//!
//! ```
//! # #[tokio::main]
//! # async fn main() {
//! # use std::time::Duration;
//! # use asyncerval::{Interval, IntervalNext, IntervalOptions};
//!
//! type Error = Box<dyn std::error::Error + Send + Sync>;
//!
//! Interval::<_, Error>::builder()
//!     // Configure frequency, can also install start, end, and error handlers here
//!     .options(IntervalOptions {
//!         frequency: Duration::from_secs(60),
//!         ..IntervalOptions::default()
//!     })
//!     // Install your event. This is a closure will be called each tick to create a new event
//!     .event(|_| Box::pin(async {
//!         // Do stuff here
//!         Ok(IntervalNext::Continue)
//!     }))
//!     .spawn(&());
//! # }
//! ```

#![warn(
    missing_docs,
    elided_lifetimes_in_paths,
    explicit_outlives_requirements,
    missing_abi,
    noop_method_call,
    pointer_structural_match,
    semicolon_in_expressions_from_macros,
    unused_import_braces,
    unused_lifetimes,
    unsafe_op_in_unsafe_fn,
    clippy::cargo,
    clippy::missing_panics_doc,
    clippy::doc_markdown,
    clippy::ptr_as_ptr,
    clippy::cloned_instead_of_copied,
    clippy::unreadable_literal,
    clippy::undocumented_unsafe_blocks,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
)]
#![forbid(unsafe_code)]

use std::future::Future;
use std::ops::Deref;
use std::pin::Pin;
use std::time::Duration;
use tokio::task::JoinError;
use tokio::time;

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
type Event<T, E> = for<'a> fn(&'a <T as Deref>::Target) -> BoxFuture<'a, Result<IntervalNext, E>>;

/// What to do when a tick is missed due to the task running longer than the specified duration
#[derive(Copy, Clone)]
pub enum MissedTickBehavior {
    /// Run all missed ticks immediately, until caught up, then resume on the original delay
    Burst,
    /// Don't run any missed ticks, but offset when ticks are run to match when the last tick ended
    Delay,
    /// Skip missed ticks, then resume on the original delay
    Skip,
}

impl From<MissedTickBehavior> for time::MissedTickBehavior {
    fn from(value: MissedTickBehavior) -> time::MissedTickBehavior {
        match value {
            MissedTickBehavior::Burst => time::MissedTickBehavior::Burst,
            MissedTickBehavior::Delay => time::MissedTickBehavior::Delay,
            MissedTickBehavior::Skip => time::MissedTickBehavior::Skip,
        }
    }
}

/// What to do on future ticks
#[derive(Default)]
pub enum IntervalNext {
    /// Continue tick execution
    #[default]
    Continue,
    /// Skip the next N ticks
    Skip(u64),
    /// Stop executing the interval
    Stop,
}

/// Configuration options for the newly created interval
#[derive(Clone)]
pub struct IntervalOptions<T, E>
where
    T: Deref,
{
    /// What to do when ticks are missed due to one taking longer than the delay
    pub on_missed: MissedTickBehavior,
    /// How often to run the interval
    pub frequency: Duration,
    /// Execute once on start to setup the interval
    pub on_start: fn(&T::Target) -> BoxFuture<'_, ()>,
    /// Execute once on end to teardown the interval
    ///
    /// This isn't guaranteed to execute - if the interval is cancelled by an external actor,
    /// this may be skipped.
    pub on_end: fn(&T::Target) -> BoxFuture<'_, ()>,
    /// Execute on an error being returned from the interval. The default handler does nothing and
    /// continues execution.
    pub on_error: fn(&T::Target, E) -> BoxFuture<'_, IntervalNext>,
    // Hack to allow struct creation syntax while allowing new variants being created
    #[doc(hidden)]
    pub __non_exhaustive: (),
}

impl<T, E> Default for IntervalOptions<T, E>
where
    T: Deref,
{
    fn default() -> Self {
        IntervalOptions {
            on_missed: MissedTickBehavior::Skip,
            frequency: Duration::from_secs(60),
            on_start: |_| Box::pin(async {}),
            on_end: |_| Box::pin(async {}),
            on_error: |_, _| Box::pin(async { IntervalNext::default() }),
            __non_exhaustive: (),
        }
    }
}

/// An interval that can be spawned and
pub struct Interval<T, E>
where
    T: Deref,
{
    data: T,
    options: IntervalOptions<T, E>,
    event: Event<T, E>,
}

impl<T, E> Interval<T, E>
where
    T: Deref + Send + Sync + 'static,
    T::Target: Send + Sync + 'static,
    E: Send + Sync + 'static,
{
    /// Create an interval builder
    pub fn builder() -> IntervalBuilder<T, E> {
        IntervalBuilder::new()
    }

    /// Spawn an interval and return the handle to await its completion
    pub fn spawn(self) -> impl Future<Output = Result<(), JoinError>> {
        let task = async move {
            let opts = self.options;

            let mut interval = tokio::time::interval(opts.frequency);

            (opts.on_start)(&self.data).await;

            let mut next = IntervalNext::Continue;
            loop {
                interval.tick().await;

                match next {
                    IntervalNext::Continue | IntervalNext::Skip(0) => {
                        let res = (self.event)(&self.data).await;
                        next = match res {
                            Ok(next) => next,
                            Err(e) => (opts.on_error)(&self.data, e).await,
                        };
                    }
                    IntervalNext::Skip(val) => {
                        next = IntervalNext::Skip(val - 1)
                    }
                    IntervalNext::Stop => {
                        break;
                    }
                }
            }

            (opts.on_end)(&self.data).await;
        };

        tokio::task::spawn(task)
    }
}

/// Builder to create and configure an interval
pub struct IntervalBuilder<T, E>
where
    T: Deref,
{
    options: Option<IntervalOptions<T, E>>,
    event: Option<Event<T, E>>,
}

impl<T, E> IntervalBuilder<T, E>
where
    T: Deref + Send + Sync + 'static,
    T::Target: Send + Sync + 'static,
    E: Send + Sync + 'static,
{
    fn new() -> Self {
        IntervalBuilder {
            options: None,
            event: None,
        }
    }

    /// Set the configuration options for this interval
    pub fn options(mut self, opts: IntervalOptions<T, E>) -> Self {
        self.options = Some(opts);
        self
    }

    /// Set the event for this interval
    pub fn event(mut self, event: Event<T, E>) -> Self {
        self.event = Some(event);
        self
    }

    /// Complete the interval and return it
    ///
    /// # Panics
    ///
    /// If a required field on the builder is missing
    pub fn build(self, data: T) -> Interval<T, E> {
        Interval {
            data,
            options: self.options.unwrap_or_default(),
            event: self.event.unwrap(),
        }
    }

    /// Complete the interval and spawn it, returning the handle to await its completion
    pub fn spawn(self, data: T) -> impl Future<Output = Result<(), JoinError>> {
        self.build(data).spawn()
    }
}
