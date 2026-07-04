use backend::{Backend, Dtype};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use utils::safetensors::{dtype_size, load_safetensors};

pub struct Arena<D: Dtype> {
	ptr: RefCell<Option<*mut u8>>,
	offset: RefCell<usize>,
	entries: RefCell<HashMap<String, (usize, Vec<usize>)>>,
	pub backend: Arc<Backend<D>>,
}

impl<D: Dtype> Arena<D> {
	pub fn new(backend: Arc<Backend<D>>) -> Self {
		Arena {
			ptr: RefCell::new(None),
			offset: RefCell::new(0),
			entries: RefCell::new(HashMap::new()),
			backend,
		}
	}

	// ROPE 固定分配f32, 如果按照D::SIZE分配, 在bf16模型上就出问题了
	pub fn alloc_bytes(&self, name: String, shape: Vec<usize>, elem_size: usize) {
		let size = (elem_size * shape.iter().product::<usize>() + 255) & !255;
		self.entries
			.borrow_mut()
			.insert(name, (*self.offset.borrow(), shape));
		*self.offset.borrow_mut() += size
	}

	pub fn alloc(&self, name: String, shape: Vec<usize>) {
		let numel: usize = shape.iter().product();
		let size = (numel * D::SIZE + 255) & !255;
		self.entries
			.borrow_mut()
			.insert(name, (*self.offset.borrow(), shape));
		*self.offset.borrow_mut() += size
	}

	pub fn shape(&self, name: &str) -> Vec<usize> {
		self.entries.borrow().get(name).unwrap().1.clone()
	}

	pub fn finalize(&self) {
		*self.ptr.borrow_mut() = Some(self.backend.device.alloc(*self.offset.borrow()));
	}

	pub fn get(&self, name: &str) -> *mut u8 {
		let ptr = self.ptr.borrow().unwrap();
		let offset = self.entries.borrow().get(name).unwrap().0;
		unsafe { ptr.add(offset) }
	}

	pub fn load_weight(&self, path: &str, skip: impl Fn(&str) -> bool) -> usize {
		let infos = load_safetensors(path).unwrap();
		let bytes = std::fs::read(path).unwrap();
		let entries = self.entries.borrow();
		let mut hit = 0;
		for info in &infos {
			if skip(&info.name) {
				continue;
			}
			let (_, shape) = entries
				.get(&info.name)
				.unwrap_or_else(|| panic!("文件 tensor {} 在 arena 中不存在", info.name));
			assert_eq!(shape, &info.shape, "{} 两边 shape 不一致", info.name);
			assert_eq!(
				dtype_size(&info.dtype),
				Some(D::SIZE),
				"{} dtype 不匹配",
				info.name
			);
			let src = &bytes[info.start..info.end];
			self.backend.device.copy_from_host_to_device(
				self.get(&info.name),
				src.as_ptr(),
				src.len(),
			);
			hit += 1;
		}
		hit
	}
}

impl<D: Dtype> Drop for Arena<D> {
	fn drop(&mut self) {
		if let Some(ptr) = *self.ptr.borrow() {
			self.backend.device.free(ptr);
		}
	}
}
