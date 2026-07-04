use backend::Dtype;
use std::marker::PhantomData;

pub struct Tensor<D: Dtype> {
	ptr: *mut u8,
	shape: Vec<usize>,
	_dtype: PhantomData<D>,
}

impl<D: Dtype> Tensor<D> {
	pub fn new(ptr: *mut u8, shape: Vec<usize>) -> Self {
		Tensor {
			ptr,
			shape,
			_dtype: PhantomData,
		}
	}

	pub fn as_ptr(&self) -> *const u8 {
		self.ptr as *const u8
	}
	pub fn as_mut_ptr(&self) -> *mut u8 {
		self.ptr
	}

	pub fn numel(&self) -> usize {
		self.shape.iter().product()
	}

	pub fn shape(&self) -> &[usize] {
		&self.shape
	}

	pub fn view(&self, shape: Vec<usize>) -> Tensor<D> {
		debug_assert!(shape.iter().product::<usize>() <= self.numel());
		Tensor::new(self.ptr, shape)
	}

	pub fn slice_rows(&self, start: usize, len: usize) -> Tensor<D> {
		debug_assert_eq!(self.shape().len(), 2);
		debug_assert!(start + len <= self.shape()[0]);
		let width = self.shape()[1];
		let ptr = unsafe {self.ptr.add(start * width * D::SIZE)};
		Tensor::new(ptr, vec![len, width])
	}
}
