import package::commands::load::cmd_load;

// Programs have to be stored in an immutable buffer because
// otherwise control flow which depends on the value of
// instructions would not be considered uniform by WGSL.

@group(1) @binding(0)
var<storage, read> instructions: array<u32>;

// This function is the virtual machine entry point. The 'lane'
// parameter needs to be set to the current lane within the
// workgroup for the purpose of selecting the correct registers.
// 'offset' is the start of the instruction sequence in the
// instructions buffer to execute. The position is the flat
// index into the shape of the program that is supposed to be
// launched. Every program, be it a prologue or epilogue,
// operates on some shape defined by the main loop. This is
// simply the shape of the tensor in the I/O register (0).

const CMD_EXIT: u32 = 0u;
const CMD_LOAD: u32 = 1u;

fn launch(lane: u32, offset: u32, position: u32) {
    var ip = offset;
    
    loop {
        switch (nextU32(&ip) & 0xFFu) {
            case CMD_LOAD: { cmd_load(lane, &ip, position); }
            case CMD_EXIT, default: { return; } 
        }
    } 
}

fn currentU32(ip: ptr<function, u32>) -> u32 {
    return instructions[*ip - 1u];
}

fn nextU32(ip: ptr<function, u32>) -> u32 {
    let value = instructions[*ip];
    *ip += 1;
    return value;
}