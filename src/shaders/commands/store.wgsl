import package::interpreter::{
    currentU32,
    nextU32
};

import package::memory::{
    store4,
    store8,
    store16,
    store32,
};

fn cmdStore(lane: u32, ip: ptr<function, u32>, position: u32) {
    let command = currentU32(ip);
    let reg = (command >> 8u) & 0xFFu;
    let bits = (command >> 16u) & 0xFFu;
    let tensor = nextU32(ip);

    switch (bits) {
        case 4: {            
            store4(lane, reg, tensor, position); 
        }
        case 8: {
            store8(lane, reg, tensor, position); 
        }
        case 16: {
            store16(lane, reg, tensor, position); 
        }
        case 32: {
            store32(lane, reg, tensor, position); 
        }
        default: {
            return;
        }
    }
}