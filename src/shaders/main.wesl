import package::interpreter::launch;

override WORKGROUP_SIZE: u32;
override REGISTER_COUNT: u32;

// This binding contains the ranges of workgroups assigned to
// a specific task. That way any workgroup can quickly
// figure out where in the wave it is assigned. The very first
// entry is special and contains 0 for min and the total number
// of tasks in max. This binding uses a dynamic offset on
// the cpu side and therefore arrayLength won't always just
// contain one wave.

@group(1) @binding(1)
var<storage, read> taskRanges: array<Range>;

struct Range {
    min: u32,
    max: u32
}

struct Builtin {
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) workgroupId: vec3u,
    @builtin(num_workgroups) numWorkgroups: vec3u
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(builtin: Builtin) {
    var workgroupIndex = builtin.workgroupId.x +
    builtin.numWorkgroups.x * (
        builtin.workgroupId.y +
        builtin.numWorkgroups.y *
        builtin.workgroupId.z
    );
    
    let taskIndex = taskIndex(&workgroupIndex);
    launch(builtin.lane, 0u, builtin.lane); 
}

fn taskIndex(workgroupIndex: ptr<function, u32>) -> u32 {
    var max = taskRanges[0].max + 1;
    var min = 1u;
    
    while (min + 1u < max) {
        let mid = min + ((max - min) >> 1u);
        if (taskRanges[mid].min <= * workgroupIndex) {
            min = mid;  
        } else {
            max = mid;
        }
    }

    *workgroupIndex -= taskRanges[min].min;
    return min;
}