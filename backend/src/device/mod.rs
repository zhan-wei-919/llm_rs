mod cuda;

pub(crate) use cuda::CudaDevice;

pub trait Device {
	fn alloc(&self, size: usize) -> *mut u8;
	fn free(&self, ptr: *mut u8);
	fn copy_from_device_to_device(&self, dst: *mut u8, src: *const u8, size: usize);
	fn copy_from_device_to_host(&self, dst: *mut u8, src: *const u8, size: usize);
	fn copy_from_host_to_device(&self, dst: *mut u8, src: *const u8, size: usize);
	fn memset(&self, dst: *mut u8, value: i32, size: usize);
	fn synchronize(&self);
}
