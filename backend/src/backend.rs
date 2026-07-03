use crate::device::{CudaDevice, Device};
use crate::dtype::Dtype;
use crate::ops::{CudaOps, Ops};
use std::marker::PhantomData;
use std::sync::Arc;

pub struct Backend<D: Dtype> {
	pub device: Arc<dyn Device>,
	pub ops: Arc<dyn Ops<D>>,
	_dtype: PhantomData<D>,
}

impl<D: Dtype> Backend<D>
where
	CudaOps: Ops<D>,
{
	pub fn cuda() -> Self {
		Backend {
			device: Arc::new(CudaDevice),
			ops: Arc::new(CudaOps),
			_dtype: PhantomData,
		}
	}
}
