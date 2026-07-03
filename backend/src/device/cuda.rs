use crate::device::Device;

pub struct CudaDevice;

impl Device for CudaDevice {
	fn alloc(&self, size: usize) -> *mut u8 {
		let mut ptr: *mut u8 = std::ptr::null_mut();
		unsafe {
			kernel::cuda::cudaMalloc(&mut ptr, size);
		}
		ptr
	}

	fn copy_from_device_to_device(&self, dst: *mut u8, src: *const u8, size: usize) {
		unsafe {
			kernel::cuda::cudaMemcpy(dst, src, size, 3);
		}
	}

	fn copy_from_device_to_host(&self, dst: *mut u8, src: *const u8, size: usize) {
		unsafe {
			kernel::cuda::cudaMemcpy(dst, src, size, 2);
		}
	}

	fn copy_from_host_to_device(&self, dst: *mut u8, src: *const u8, size: usize) {
		unsafe {
			kernel::cuda::cudaMemcpy(dst, src, size, 1);
		}
	}

	fn free(&self, ptr: *mut u8) {
		unsafe {
			kernel::cuda::cudaFree(ptr);
		}
	}

	fn memset(&self, dst: *mut u8, value: i32, size: usize) {
		unsafe {
			kernel::cuda::cudaMemset(dst, value, size);
		}
	}

	fn synchronize(&self) {
		unsafe {
			kernel::cuda::cudaDeviceSynchronize();
		}
	}
}
