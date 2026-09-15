import package::interpreter::launch;
import package::mainloops::matmul::taskMatmul;

override WORKGROUP_SIZE: u32;
override REGISTER_COUNT: u32;

// This binding contains the ranges of workgroups assigned to
// a specific task. That way any workgroup can quickly
// figure out where in the wave it is assigned. The very first
// entry is special and contains 0 for min and the total number
// of tasks in max. This binding uses a dynamic offset on
// the cpu side and therefore arrayLength won't always just
// contain one wave.

@group(1) @binding(0)
var<storage, read> taskRanges: array<Range>;
@group(1) @binding(1)
var<storage, read> taskOffsets: array<u32>;
@group(1) @binding(2)
var<storage, read> tasks: array<u32>;

const TASK_EXIT: u32 = 0u;
const TASK_MATMUL: u32 = 1u;

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
    let lane = builtin.lane;
    var group = builtin.workgroupId.x +
    builtin.numWorkgroups.x * (
        builtin.workgroupId.y +
        builtin.numWorkgroups.y *
        builtin.workgroupId.z
    );
    
    let taskIndex = searchTaskIndex(&group);
    if (taskIndex == 0u) { return; } 
    var tp = taskOffsets[taskIndex - 1u];

    switch (nextU32(&tp)) {
        case TASK_MATMUL: { taskMatmul(group, lane, &tp); }
        default: {
            return;
        }
    }
}

// Returns a 1-based task index, or 0 if no task owns this
// workgroup. For a valid task, converts workgroupIndex to the
// task-local index.
fn searchTaskIndex(group: ptr<function, u32>) -> u32 {
    let taskCount = taskRanges[0].max;
    if (taskCount == 0u) { return 0u; }

    var max = taskCount + 1u;
    var min = 1u;
    
    while (min + 1u < max) {
        let mid = min + ((max - min) >> 1u);
        if (taskRanges[mid].min <= * group) {
            min = mid;  
        } else {
            max = mid;
        }
    }

    let range = taskRanges[min];
    if ((*group < range.min) || (*group >= range.max)) {
        return 0u;
    }

    *group -= range.min;
    return min;
}

fn currentU32(tp: ptr<function, u32>) -> u32 {
    return tasks[*tp - 1u];
}

fn nextU32(tp: ptr<function, u32>) -> u32 {
    let value = tasks[*tp];
    *tp += 1;
    return value;
}
