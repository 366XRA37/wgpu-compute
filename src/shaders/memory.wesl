import package::main::{WORKGROUP_SIZE, REGISTER_COUNT};

// These storage bindings contain tensors made up of 256 KiB
// pages. A tensor is a 32-bit page index, where the least
// significant 16 bits indicate the index within a memory pool
// and the following 15 bits are the index of the pool,
// currently an integer between 0 and 3. The most significant
// bit indicates if this tensor points directly to a data page
// or one filled with other page indices, if the data can't fit
// inside just one page. A 256 KiB page can fit at most u16::MAX
// other page indices, which, at the chosen page size, means
// the maximum tensor size is about 16 GiB. Since the largest
// supported data type is 32 bits wide and no tensor is allowed
// to contain more than u32::MAX elements, this fits perfectly.

@group(0) @binding(0)
var<storage, read_write> memoryA: array<u32>;

@group(0) @binding(1)
var<storage, read_write> memoryB: array<u32>;

@group(0) @binding(2)
var<storage, read_write> memoryC: array<u32>;

@group(0) @binding(3)
var<storage, read_write> memoryD: array<u32>;

var<workgroup> registers: array<
    u32, 
    WORKGROUP_SIZE * REGISTER_COUNT
>;

fn loadRegister(lane: u32, index: u32) -> u32 {
    return registers[index * WORKGROUP_SIZE + lane];
}

fn storeRegister(lane: u32, index: u32, value: u32) {
    registers[index * WORKGROUP_SIZE + lane] = value;
}

fn load4(tensor: u32, index: u32) -> u32 {
    let word = load32(tensor, index >> 3u);
    let shift = (index & 7u) << 2u;
    return (word >> shift) & 0xFu;
}

fn load8(tensor: u32, index: u32) -> u32 {
    let word = load32(tensor, index >> 2u);
    let shift = (index & 3u) << 3u;
    return (word >> shift) & 0xFFu;
}

fn load16(tensor: u32, index: u32) -> u32 {
    let word = load32(tensor, index >> 1u);
    let shift = (index & 1u) << 4u;
    return (word >> shift) & 0xFFFFu;
}

fn load32(tensor: u32, index: u32) -> u32 {
    var pageIndex = tensor;
    var elementIndex = index;

    if (bool(pageIndex >> 31u)) {
        let pageOffset = index >> 16u;
        elementIndex = index & 0xFFFFu;
        pageIndex = loadPhysical32(
            pageIndex & 0x7FFFFFFFu,
            pageOffset
        );
    }

    return loadPhysical32(pageIndex, elementIndex);
}

fn store32(tensor: u32, index: u32, value: u32) {
    var pageIndex = tensor;
    var elementIndex = index;

    if (bool(pageIndex >> 31u)) {
        let pageOffset = index >> 16u;
        elementIndex = index & 0xFFFFu;
        pageIndex = loadPhysical32(
            pageIndex & 0x7FFFFFFFu,
            pageOffset
        );
    }

    storePhysical32(pageIndex, elementIndex, value);
}

fn storePhysical32(page: u32, index: u32, value: u32) {
    let elementIndex = ((page & 0xFFFFu) << 16u) + index; 
    let poolIndex = page >> 16u;

    switch (poolIndex) {
        case 0: { memoryA[elementIndex] = value; }
        case 1: { memoryB[elementIndex] = value; }
        case 2: { memoryC[elementIndex] = value; }
        case 3: { memoryD[elementIndex] = value; }
        default: { return; }
    }
}

fn loadPhysical32(page: u32, index: u32) -> u32 {
    let elementIndex = ((page & 0xFFFFu) << 16u) + index; 
    let poolIndex = page >> 16u;

    switch (poolIndex) {
        case 0: { return memoryA[elementIndex]; }
        case 1: { return memoryB[elementIndex]; }
        case 2: { return memoryC[elementIndex]; }
        case 3: { return memoryD[elementIndex]; }
        default: { return 0u; }
    }
}
