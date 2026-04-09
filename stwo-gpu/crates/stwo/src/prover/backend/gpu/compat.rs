//! Compatibility module for cudarc 0.11+ API
//!
//! cudarc 0.11 restructured the API - raw CUDA driver functions are now
//! available through the `result` module instead of directly in `sys`.
//!
//! This module provides wrapper functions to maintain compatibility with
//! the existing GPU backend code. For functions not exposed by cudarc 0.11.9's
//! `result` module, we use direct FFI calls to the CUDA driver library.

#[cfg(feature = "cuda-runtime")]
use cudarc::driver::result;

#[cfg(feature = "cuda-runtime")]
use cudarc::driver::sys::{CUdevice, CUdevice_attribute};

// =============================================================================
// Direct FFI bindings for CUDA Driver API functions not exposed by cudarc 0.11.9
// =============================================================================

#[cfg(feature = "cuda-runtime")]
#[link(name = "cuda")]
extern "C" {
    fn cuDeviceCanAccessPeer(canAccessPeer: *mut i32, dev: i32, peerDev: i32) -> i32;
    fn cuDeviceGetP2PAttribute(value: *mut i32, attrib: i32, srcDevice: i32, dstDevice: i32) -> i32;
    fn cuCtxEnablePeerAccess(peerContext: cudarc::driver::sys::CUcontext, Flags: u32) -> i32;
    fn cuCtxSetCurrent(ctx: cudarc::driver::sys::CUcontext) -> i32;
    fn cuMemcpyPeer(
        dstDevice: u64,
        dstContext: cudarc::driver::sys::CUcontext,
        srcDevice: u64,
        srcContext: cudarc::driver::sys::CUcontext,
        ByteCount: usize,
    ) -> i32;
    fn cuMemcpyPeerAsync(
        dstDevice: u64,
        dstContext: cudarc::driver::sys::CUcontext,
        srcDevice: u64,
        srcContext: cudarc::driver::sys::CUcontext,
        ByteCount: usize,
        hStream: cudarc::driver::sys::CUstream,
    ) -> i32;
    fn cuMemcpy(dst: u64, src: u64, ByteCount: usize) -> i32;
    fn cuStreamSynchronize(hStream: cudarc::driver::sys::CUstream) -> i32;
    fn cuStreamBeginCapture(hStream: cudarc::driver::sys::CUstream, mode: i32) -> i32;
    fn cuStreamEndCapture(
        hStream: cudarc::driver::sys::CUstream,
        phGraph: *mut cudarc::driver::sys::CUgraph,
    ) -> i32;
    fn cuGraphInstantiateWithFlags(
        phGraphExec: *mut cudarc::driver::sys::CUgraphExec,
        hGraph: cudarc::driver::sys::CUgraph,
        flags: u64,
    ) -> i32;
    fn cuGraphLaunch(
        hGraphExec: cudarc::driver::sys::CUgraphExec,
        hStream: cudarc::driver::sys::CUstream,
    ) -> i32;
    fn cuGraphDestroy(hGraph: cudarc::driver::sys::CUgraph) -> i32;
    fn cuGraphExecDestroy(hGraphExec: cudarc::driver::sys::CUgraphExec) -> i32;
    fn cuMemcpyHtoDAsync_v2(dstDevice: u64, srcHost: *const std::ffi::c_void, ByteCount: usize, hStream: *mut std::ffi::c_void) -> i32;
    fn cuMemcpyDtoHAsync_v2(dstHost: *mut std::ffi::c_void, srcDevice: u64, ByteCount: usize, hStream: *mut std::ffi::c_void) -> i32;
    fn cuMemAllocHost_v2(pp: *mut *mut std::ffi::c_void, bytesize: usize) -> i32;
    fn cuMemFreeHost(p: *mut std::ffi::c_void) -> i32;
    fn cuGraphExecUpdate(
        hGraphExec: cudarc::driver::sys::CUgraphExec,
        hGraph: cudarc::driver::sys::CUgraph,
        resultInfo: *mut CUgraphExecUpdateResultInfo,
    ) -> i32;
}

/// Result info for graph exec update (matches CUDA driver struct).
/// We use raw pointers for node types to avoid depending on CUgraphNode being exported.
#[cfg(feature = "cuda-runtime")]
#[repr(C)]
#[allow(non_snake_case)]
pub struct CUgraphExecUpdateResultInfo {
    pub result: CUgraphExecUpdateResult,
    pub errorNode: *mut std::ffi::c_void,
    pub errorFromNode: *mut std::ffi::c_void,
}

/// CUDA_SUCCESS constant (value 0).
#[cfg(feature = "cuda-runtime")]
const CUDA_SUCCESS: i32 = 0;

/// CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED (value 704).
#[cfg(feature = "cuda-runtime")]
const CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED: i32 = 704;

/// Query available and total GPU memory.
#[cfg(feature = "cuda-runtime")]
pub fn mem_get_info() -> Result<(usize, usize), String> {
    result::mem_get_info()
        .map_err(|e| format!("Failed to query memory info: {:?}", e))
}

/// Query a device attribute.
#[cfg(feature = "cuda-runtime")]
pub fn device_get_attribute(device: CUdevice, attrib: CUdevice_attribute) -> Result<i32, String> {
    unsafe { result::device::get_attribute(device, attrib) }
        .map_err(|e| format!("Failed to query device attribute: {:?}", e))
}

/// Query device name.
#[cfg(feature = "cuda-runtime")]
pub fn device_get_name(device: CUdevice) -> Result<String, String> {
    result::device::get_name(device)
        .map_err(|e| format!("Failed to query device name: {:?}", e))
}

/// Query total device memory.
#[cfg(feature = "cuda-runtime")]
pub fn device_total_mem(device: CUdevice) -> Result<usize, String> {
    unsafe { result::device::total_mem(device) }
        .map_err(|e| format!("Failed to query total memory: {:?}", e))
}

/// Get device count.
#[cfg(feature = "cuda-runtime")]
pub fn device_get_count() -> Result<i32, String> {
    result::device::get_count()
        .map_err(|e| format!("Failed to query device count: {:?}", e))
}

/// Get device by ordinal.
#[cfg(feature = "cuda-runtime")]
pub fn device_get(ordinal: i32) -> Result<CUdevice, String> {
    result::device::get(ordinal)
        .map_err(|e| format!("Failed to get device: {:?}", e))
}

// =============================================================================
// Pinned Memory Functions
// =============================================================================

/// Allocate pinned (page-locked) host memory.
/// Returns a raw pointer that can be used for async GPU transfers.
#[cfg(feature = "cuda-runtime")]
pub fn mem_alloc_host(size_bytes: usize) -> Result<*mut std::ffi::c_void, String> {
    let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let result = unsafe { cuMemAllocHost_v2(&mut ptr, size_bytes) };
    if result == CUDA_SUCCESS {
        Ok(ptr)
    } else {
        Err(format!("Failed to allocate pinned memory: CUDA error {}", result))
    }
}

/// Free pinned host memory.
#[cfg(feature = "cuda-runtime")]
pub fn mem_free_host(ptr: *mut std::ffi::c_void) -> Result<(), String> {
    let result = unsafe { cuMemFreeHost(ptr) };
    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("Failed to free pinned memory: CUDA error {}", result))
    }
}

// =============================================================================
// Async Memory Copy Functions
// =============================================================================

/// Async host-to-device memory copy.
#[cfg(feature = "cuda-runtime")]
pub fn memcpy_htod_async(
    dst: cudarc::driver::sys::CUdeviceptr,
    src: *const std::ffi::c_void,
    size_bytes: usize,
    stream: cudarc::driver::sys::CUstream,
) -> Result<(), String> {
    let result = unsafe {
        cuMemcpyHtoDAsync_v2(dst, src, size_bytes, stream as *mut std::ffi::c_void)
    };
    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("Async H2D copy failed: CUDA error {}", result))
    }
}

/// Async device-to-host memory copy.
#[cfg(feature = "cuda-runtime")]
pub fn memcpy_dtoh_async(
    dst: *mut std::ffi::c_void,
    src: cudarc::driver::sys::CUdeviceptr,
    size_bytes: usize,
    stream: cudarc::driver::sys::CUstream,
) -> Result<(), String> {
    let result = unsafe {
        cuMemcpyDtoHAsync_v2(dst, src as u64, size_bytes, stream as *mut std::ffi::c_void)
    };
    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("Async D2H copy failed: CUDA error {}", result))
    }
}

// =============================================================================
// P2P (Peer-to-Peer) Access Functions
// =============================================================================

/// Check if one device can access memory on another device.
#[cfg(feature = "cuda-runtime")]
pub fn device_can_access_peer(src_device: i32, dst_device: i32) -> Result<bool, String> {
    let mut can_access: i32 = 0;
    let result = unsafe {
        cuDeviceCanAccessPeer(&mut can_access, src_device, dst_device)
    };

    if result == CUDA_SUCCESS {
        Ok(can_access != 0)
    } else {
        Err(format!("Failed to query P2P access: CUDA error {}", result))
    }
}

/// Get P2P attribute between two devices.
#[cfg(feature = "cuda-runtime")]
pub fn device_get_p2p_attribute(
    attrib: CUdevice_P2PAttribute,
    src_device: i32,
    dst_device: i32,
) -> Result<i32, String> {
    let mut value: i32 = 0;
    let result = unsafe {
        cuDeviceGetP2PAttribute(&mut value, attrib as i32, src_device, dst_device)
    };

    if result == CUDA_SUCCESS {
        Ok(value)
    } else {
        Err(format!("Failed to query P2P attribute: CUDA error {}", result))
    }
}

/// Enable P2P access from current context to peer context.
#[cfg(feature = "cuda-runtime")]
pub fn ctx_enable_peer_access(
    peer_ctx: cudarc::driver::sys::CUcontext,
    flags: u32,
) -> Result<(), String> {
    let result = unsafe {
        cuCtxEnablePeerAccess(peer_ctx, flags)
    };

    if result == CUDA_SUCCESS || result == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED {
        Ok(())
    } else {
        Err(format!("Failed to enable P2P access: CUDA error {}", result))
    }
}

/// Set current CUDA context.
#[cfg(feature = "cuda-runtime")]
pub fn ctx_set_current(ctx: cudarc::driver::sys::CUcontext) -> Result<(), String> {
    let result = unsafe {
        cuCtxSetCurrent(ctx)
    };

    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("Failed to set current context: CUDA error {}", result))
    }
}

// =============================================================================
// Memory Copy Functions
// =============================================================================

/// Synchronous peer-to-peer memory copy between GPUs.
#[cfg(feature = "cuda-runtime")]
pub fn memcpy_peer(
    dst: cudarc::driver::sys::CUdeviceptr,
    dst_ctx: cudarc::driver::sys::CUcontext,
    src: cudarc::driver::sys::CUdeviceptr,
    src_ctx: cudarc::driver::sys::CUcontext,
    size_bytes: usize,
) -> Result<(), String> {
    let result = unsafe {
        cuMemcpyPeer(dst, dst_ctx, src, src_ctx, size_bytes)
    };

    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("P2P memory copy failed: CUDA error {}", result))
    }
}

/// Asynchronous peer-to-peer memory copy between GPUs.
#[cfg(feature = "cuda-runtime")]
pub fn memcpy_peer_async(
    dst: cudarc::driver::sys::CUdeviceptr,
    dst_ctx: cudarc::driver::sys::CUcontext,
    src: cudarc::driver::sys::CUdeviceptr,
    src_ctx: cudarc::driver::sys::CUcontext,
    size_bytes: usize,
    stream: cudarc::driver::sys::CUstream,
) -> Result<(), String> {
    let result = unsafe {
        cuMemcpyPeerAsync(dst, dst_ctx, src, src_ctx, size_bytes, stream)
    };

    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("Async P2P memory copy failed: CUDA error {}", result))
    }
}

/// Synchronous device-to-device memory copy (same GPU).
#[cfg(feature = "cuda-runtime")]
pub fn memcpy_dtod(
    dst: cudarc::driver::sys::CUdeviceptr,
    src: cudarc::driver::sys::CUdeviceptr,
    size_bytes: usize,
) -> Result<(), String> {
    let result = unsafe {
        cuMemcpy(dst, src, size_bytes)
    };

    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("D2D memory copy failed: CUDA error {}", result))
    }
}

// =============================================================================
// Stream Synchronization
// =============================================================================

/// Synchronize a CUDA stream, waiting for all operations to complete.
#[cfg(feature = "cuda-runtime")]
pub fn stream_synchronize(stream: cudarc::driver::sys::CUstream) -> Result<(), String> {
    let result = unsafe {
        cuStreamSynchronize(stream)
    };

    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("Stream synchronization failed: CUDA error {}", result))
    }
}

// =============================================================================
// CUDA Graph Functions
// =============================================================================

/// Begin stream capture for CUDA graph creation.
#[cfg(feature = "cuda-runtime")]
pub fn stream_begin_capture(
    stream: cudarc::driver::sys::CUstream,
    mode: CUstreamCaptureMode,
) -> Result<(), String> {
    let result = unsafe {
        cuStreamBeginCapture(stream, mode as i32)
    };

    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("Stream capture begin failed: CUDA error {}", result))
    }
}

/// End stream capture and get the graph.
#[cfg(feature = "cuda-runtime")]
pub fn stream_end_capture(
    stream: cudarc::driver::sys::CUstream,
) -> Result<cudarc::driver::sys::CUgraph, String> {
    let mut graph: cudarc::driver::sys::CUgraph = std::ptr::null_mut();
    let result = unsafe {
        cuStreamEndCapture(stream, &mut graph)
    };

    if result == CUDA_SUCCESS && !graph.is_null() {
        Ok(graph)
    } else {
        Err(format!("Stream capture end failed: CUDA error {}", result))
    }
}

/// Instantiate a graph for execution.
#[cfg(feature = "cuda-runtime")]
pub fn graph_instantiate(
    graph: cudarc::driver::sys::CUgraph,
) -> Result<cudarc::driver::sys::CUgraphExec, String> {
    let mut graph_exec: cudarc::driver::sys::CUgraphExec = std::ptr::null_mut();
    let result = unsafe {
        cuGraphInstantiateWithFlags(
            &mut graph_exec,
            graph,
            0, // flags
        )
    };

    if result == CUDA_SUCCESS {
        Ok(graph_exec)
    } else {
        Err(format!("Graph instantiation failed: CUDA error {}", result))
    }
}

/// Launch an instantiated graph.
#[cfg(feature = "cuda-runtime")]
pub fn graph_launch(
    graph_exec: cudarc::driver::sys::CUgraphExec,
    stream: cudarc::driver::sys::CUstream,
) -> Result<(), String> {
    let result = unsafe {
        cuGraphLaunch(graph_exec, stream)
    };

    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("Graph launch failed: CUDA error {}", result))
    }
}

/// Destroy a graph.
#[cfg(feature = "cuda-runtime")]
pub fn graph_destroy(graph: cudarc::driver::sys::CUgraph) -> Result<(), String> {
    let result = unsafe {
        cuGraphDestroy(graph)
    };

    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("Graph destroy failed: CUDA error {}", result))
    }
}

/// Destroy a graph exec.
#[cfg(feature = "cuda-runtime")]
pub fn graph_exec_destroy(graph_exec: cudarc::driver::sys::CUgraphExec) -> Result<(), String> {
    let result = unsafe {
        cuGraphExecDestroy(graph_exec)
    };

    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("Graph exec destroy failed: CUDA error {}", result))
    }
}

/// Update a graph exec in place.
/// Returns the update result.
#[cfg(feature = "cuda-runtime")]
pub fn graph_exec_update(
    graph_exec: cudarc::driver::sys::CUgraphExec,
    graph: cudarc::driver::sys::CUgraph,
) -> Result<CUgraphExecUpdateResult, String> {
    let mut result_info = CUgraphExecUpdateResultInfo {
        result: CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_SUCCESS,
        errorNode: std::ptr::null_mut() as *mut std::ffi::c_void,
        errorFromNode: std::ptr::null_mut() as *mut std::ffi::c_void,
    };

    let result = unsafe {
        cuGraphExecUpdate(graph_exec, graph, &mut result_info)
    };

    if result == CUDA_SUCCESS {
        Ok(result_info.result)
    } else {
        Err(format!("Graph exec update failed: CUDA error {}", result))
    }
}

// Re-export common types for convenience
#[cfg(feature = "cuda-runtime")]
pub use cudarc::driver::sys::{
    CUdevice_attribute as CUdevice_attribute_enum,
    CUresult as CudaResult,
    CUstream,
    CUdeviceptr,
    CUcontext,
    CUdevice_P2PAttribute,
    CUgraph,
    CUgraphExec,
    CUstreamCaptureMode,
};

/// Graph exec update result enum.
/// Defined here since cudarc 0.11.9 may not export it directly.
#[cfg(feature = "cuda-runtime")]
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum CUgraphExecUpdateResult {
    CU_GRAPH_EXEC_UPDATE_SUCCESS = 0,
    CU_GRAPH_EXEC_UPDATE_ERROR = 1,
    CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED = 2,
    CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED = 3,
    CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED = 4,
    CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED = 5,
    CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED = 6,
    CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE = 7,
    CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED = 8,
}
