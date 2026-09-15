import package::interpreter::{
    currentU32,
    nextU32
};

import package::memory::{
    load4,
    load8,
    load16,
    load32,
    storeRegister
};

fn cmdLoad(lane: u32, ip: ptr<function, u32>, position: u32) {
    let command = currentU32(ip);
    let reg = (command >> 8u) & 0xFFu;
    let bits = (command >> 16u) & 0xFFu;
    let tensor = nextU32(ip);
    var value = 0u;

    switch (bits) {
        case 4: {
            value = load4(tensor, position); 
        }
        case 8: {
            value = load8(tensor, position); 
        }
        case 16: {
            value = load16(tensor, position); 
        }
        case 32: {
            value = load32(tensor, position); 
        }
        default: {
            value = 0u;
        }
    }

    storeRegister(lane, reg, value);
}