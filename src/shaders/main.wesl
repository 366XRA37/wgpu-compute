import package::interpreter::launch;

override WORKGROUP_SIZE: u32;
override REGISTER_COUNT: u32;

struct Builtin {
    @builtin(local_invocation_index) lane: u32
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(builtin: Builtin) {
    launch(builtin.lane, 0u, builtin.lane);
}