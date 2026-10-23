//! Logs as the Python package prints them: `LEVEL: message` on stderr, for this crate only.

struct Logger {
    level: log::LevelFilter,
}

impl log::Log for Logger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= self.level && metadata.target().starts_with("pocket_tts")
    }

    fn log(&self, record: &log::Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let level = match record.level() {
            log::Level::Error => "ERROR",
            log::Level::Warn => "WARNING",
            log::Level::Info => "INFO",
            log::Level::Debug => "DEBUG",
            log::Level::Trace => "TRACE",
        };
        eprintln!("{level}: {}", record.args());
    }

    fn flush(&self) {}
}

/// `quiet` keeps the errors only.
pub fn init(quiet: bool) {
    let level = if quiet {
        log::LevelFilter::Error
    } else {
        log::LevelFilter::Info
    };
    let _ = log::set_boxed_logger(Box::new(Logger { level }));
    log::set_max_level(level);
}
