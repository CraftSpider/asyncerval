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
use time::OffsetDateTime;
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior as TokioMTB;

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
type Event<T, E> = for<'a> fn(&'a <T as Deref>::Target) -> BoxFuture<'a, Result<IntervalNext, E>>;

/// What to do when a tick is missed due to the task running longer than the specified duration
#[derive(Copy, Clone, Default)]
#[non_exhaustive]
pub enum MissedTickBehavior {
    /// Run all missed ticks immediately, until caught up, then resume on the original delay
    Burst,
    /// Don't run any missed ticks, but offset when ticks are run to match when the last tick ended
    Delay,
    /// Skip missed ticks, then resume on the original delay
    #[default]
    Skip,
}

impl From<MissedTickBehavior> for TokioMTB {
    fn from(value: MissedTickBehavior) -> TokioMTB {
        match value {
            MissedTickBehavior::Burst => TokioMTB::Burst,
            MissedTickBehavior::Delay => TokioMTB::Delay,
            MissedTickBehavior::Skip => TokioMTB::Skip,
        }
    }
}

/// What to do on future ticks
#[derive(Clone, Default)]
#[non_exhaustive]
pub enum IntervalNext {
    /// Continue tick execution
    #[default]
    Continue,
    /// Skip the next N ticks
    Skip(u64),
    /// Stop executing the interval
    Stop,
}

/// Align a task to start at a specific time boundary
#[derive(Copy, Clone, Default, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum AlignTo {
    /// Don't align the task at all, start immediately
    #[default]
    None,
    /// Align to the nearest minute
    Minute,
    /// Align to the nearest hour
    Hour,
    /// Align to the nearest day
    Day,
}

impl AlignTo {
    fn align(self, date_time: OffsetDateTime) -> OffsetDateTime {
        let mut time = date_time.time();
        let mut dur = Duration::default();
        if self >= AlignTo::Minute {
            let sec = time.second();
            if sec > 0 {
                let new_dur = Duration::from_secs((60 - sec) as u64);
                dur += new_dur;
                time += new_dur;
            }
        }
        if self >= AlignTo::Hour {
            let minute = time.minute();
            if minute > 0 {
                let new_dur = Duration::from_secs((60 - minute) as u64 * 60);
                dur += new_dur;
                time += new_dur;
            }
        }
        if self >= AlignTo::Day {
            let hour = time.hour();
            if hour > 0 {
                let new_dur = Duration::from_secs((24 - hour) as u64 * 3600);
                dur += new_dur;
                time += new_dur;
            }
        }
        date_time + dur
    }
}

/// Configuration options for the newly created interval
#[derive(Clone)]
pub struct IntervalOptions<T, E>
where
    T: Deref,
{
    /// How often to run the interval
    pub frequency: Duration,
    /// Align interval to start aligned to a specific unit of time - the nearest second/minute/hour
    pub align_to: AlignTo,
    /// What to do when ticks are missed due to one taking longer than the delay
    pub on_missed: MissedTickBehavior,

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
            on_missed: MissedTickBehavior::default(),
            align_to: AlignTo::default(),
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
    pub fn spawn(self) -> JoinHandle<()> {
        let task = async move {
            let opts = self.options;

            (opts.on_start)(&self.data).await;

            let now = OffsetDateTime::now_utc();
            let then = opts.align_to.align(now);
            if then > now {
                tokio::time::sleep((then - now).unsigned_abs()).await;
            }

            let mut interval = tokio::time::interval(opts.frequency);
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
    pub fn spawn(self, data: T) -> JoinHandle<()> {
        self.build(data).spawn()
    }
}

#[cfg(test)]
mod tests {
    use time::{Date, Month, PrimitiveDateTime, Time, UtcOffset};
    use super::*;

    fn dt_from(date: Date, time: Time) -> OffsetDateTime {
        PrimitiveDateTime::new(date, time).assume_offset(UtcOffset::UTC)
    }

    #[test]
    fn test_align_none() {
        let a = AlignTo::None;

        let date = Date::from_calendar_date(2022, Month::January, 5).unwrap();
        let time = Time::from_hms(1, 1, 1).unwrap();
        assert_eq!(
            a.align(dt_from(date, time)),
            dt_from(date, time),
        );
    }

    #[test]
    fn test_align_minute() {
        let a = AlignTo::Minute;

        let date = Date::from_calendar_date(2022, Month::January, 5).unwrap();
        let time1 = Time::from_hms(0, 0, 0).unwrap();
        assert_eq!(
            a.align(dt_from(date, time1)),
            dt_from(date, time1),
        );

        let time2 = Time::from_hms(0, 0, 30).unwrap();
        let res_time2 = Time::from_hms(0, 1, 0).unwrap();
        assert_eq!(
            a.align(dt_from(date, time2)),
            dt_from(date, res_time2),
        );

        let time3 = Time::from_hms(0, 59, 30).unwrap();
        let res_time3 = Time::from_hms(1, 0, 0).unwrap();
        assert_eq!(
            a.align(dt_from(date, time3)),
            dt_from(date, res_time3),
        );

        let time4 = Time::from_hms(23, 59, 30).unwrap();
        let res_time4 = Time::from_hms(0, 0, 0).unwrap();
        assert_eq!(
            a.align(dt_from(date, time4)),
            dt_from(date.next_day().unwrap(), res_time4),
        );
    }

    #[test]
    fn test_align_hour() {
        let a = AlignTo::Hour;

        let date = Date::from_calendar_date(2022, Month::January, 5).unwrap();
        let time1 = Time::from_hms(0, 0, 0).unwrap();
        assert_eq!(
            a.align(dt_from(date, time1)),
            dt_from(date, time1),
        );

        let time2 = Time::from_hms(0, 0, 30).unwrap();
        let res_time2 = Time::from_hms(1, 0, 0).unwrap();
        assert_eq!(
            a.align(dt_from(date, time2)),
            dt_from(date, res_time2),
        );

        let time3 = Time::from_hms(0, 59, 30).unwrap();
        let res_time3 = Time::from_hms(1, 0, 0).unwrap();
        assert_eq!(
            a.align(dt_from(date, time3)),
            dt_from(date, res_time3),
        );

        let time4 = Time::from_hms(23, 59, 30).unwrap();
        let res_time4 = Time::from_hms(0, 0, 0).unwrap();
        assert_eq!(
            a.align(dt_from(date, time4)),
            dt_from(date.next_day().unwrap(), res_time4),
        );
    }

    #[test]
    fn test_align_day() {
        let a = AlignTo::Day;

        let date = Date::from_calendar_date(2022, Month::January, 5).unwrap();
        let time1 = Time::from_hms(0, 0, 0).unwrap();
        assert_eq!(
            a.align(dt_from(date, time1)),
            dt_from(date, time1),
        );

        let time2 = Time::from_hms(0, 0, 30).unwrap();
        let res_time2 = Time::from_hms(0, 0, 0).unwrap();
        assert_eq!(
            a.align(dt_from(date, time2)),
            dt_from(date.next_day().unwrap(), res_time2),
        );

        let time3 = Time::from_hms(0, 59, 30).unwrap();
        let res_time3 = Time::from_hms(0, 0, 0).unwrap();
        assert_eq!(
            a.align(dt_from(date, time3)),
            dt_from(date.next_day().unwrap(), res_time3),
        );

        let time4 = Time::from_hms(23, 59, 30).unwrap();
        let res_time4 = Time::from_hms(0, 0, 0).unwrap();
        assert_eq!(
            a.align(dt_from(date, time4)),
            dt_from(date.next_day().unwrap(), res_time4),
        );
    }
}
