extern crate lalrpop;

fn main() {
    lalrpop::Configuration::new()
        // By default build.rs causes the binary to recompile each time *any* file in the whole project changes.
        // This wastes time when only touching stuff like .fml files. `emit_rerun_directives` prevents that.
        .emit_rerun_directives(true)
        // Automatically find the .lalrpop file in the project dir.
        // AFAIK there's no benefit to specifying the path manually.
        // OTOH when i tried, i realized i'd ned to also manually specify the output dir
        // because, unintuitively, process_dir does some extra stuff compared to process_file.
        // Also note that `process_dir("src")` causes the output to be generated in `out/src/fml.rs`
        // instead of `out/fml.rs` which means it won't be found by `lalrpop_mod!`.
        // Even worse, if the original still exists, it'll find that so your changes won't have any effect,
        // have fun debugging that.
        // More info: https://github.com/lalrpop/lalrpop/issues/677
        .process_current_dir()
        .unwrap();
}
