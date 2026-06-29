use std::sync::Arc;
use std::marker::PhantomData;
use backend::device::Device;
use backend::dtype::Dtype;
use backend::ops::Ops;

pub struct Tensor<D: Dtype> {
	ptr: *mut u8,
	shape: Vec<usize>,
	device: Arc<dyn Device>,
	ops: Arc<dyn Ops<D>>,
	_dtype: PhantomData<D>,
}