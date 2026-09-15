use naga::valid::{Capabilities, ValidationFlags, Validator};
use wesl::Compiler;

fn main() {
    println!("cargo::rerun-if-changed=src/shaders");

    let compiler = Compiler::default();
    let compile_result = compiler
        .compile_module("src/shaders", &"package::main".parse().unwrap())
        .unwrap_or_else(|e| panic!("{e}"));

    let wgsl = compile_result.to_string();
    let module = naga::front::wgsl::parse_str(&wgsl)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&wgsl)));

    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&wgsl)));

    compile_result.write_artifact("main");
}
