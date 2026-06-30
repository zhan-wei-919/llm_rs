use backend::{Dtype, Backend};
use tensor::{Arena, Tensor};
use std::sync::Arc;

pub struct Transformer<D: Dtype> {
	ln1: Tensor<D>,
	
}