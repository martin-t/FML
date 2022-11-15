use anyhow::Result;

#[macro_export]
macro_rules! veccat {
    ($a:expr, $b:expr) => {
        $a.into_iter().chain($b.into_iter()).collect()
    };
    ($a:expr, $b:expr, $c:expr) => {
        $a.into_iter().chain($b.into_iter()).chain($c.into_iter()).collect()
    };
}
