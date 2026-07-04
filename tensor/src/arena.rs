use backend::{Backend, Dtype};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use utils::safetensors::{dtype_size, load_safetensors};

/// arena 中一块显存的句柄，由 alloc 返回或 slot_id 解析得到，
/// 之后可用 get_by_slot / shape_by_slot 免去字符串哈希查找
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SlotId(u32);

pub struct Arena<D: Dtype> {
	ptr: RefCell<Option<*mut u8>>,
	offset: RefCell<usize>,
	entries: RefCell<HashMap<String, SlotId>>,
	slots: RefCell<Vec<(usize, Vec<usize>)>>,
	pub backend: Arc<Backend<D>>,
}

impl<D: Dtype> Arena<D> {
	pub fn new(backend: Arc<Backend<D>>) -> Self {
		Arena {
			ptr: RefCell::new(None),
			offset: RefCell::new(0),
			entries: RefCell::new(HashMap::new()),
			slots: RefCell::new(Vec::new()),
			backend,
		}
	}

	pub fn alloc(&self, name: String, shape: Vec<usize>) -> SlotId {
		let numel: usize = shape.iter().product();
		let size = (numel * D::SIZE + 255) & !255;
		let mut slots = self.slots.borrow_mut();
		let id = SlotId(slots.len() as u32);
		slots.push((*self.offset.borrow(), shape));
		self.entries.borrow_mut().insert(name, id);
		*self.offset.borrow_mut() += size;
		id
	}

	pub fn slot_id(&self, name: &str) -> SlotId {
		*self
			.entries
			.borrow()
			.get(name)
			.unwrap_or_else(|| panic!("tensor {} 在 arena 中不存在", name))
	}

	pub fn shape(&self, name: &str) -> Vec<usize> {
		self.shape_by_slot(self.slot_id(name))
	}

	pub fn shape_by_slot(&self, id: SlotId) -> Vec<usize> {
		self.slots.borrow()[id.0 as usize].1.clone()
	}

	pub fn finalize(&self) {
		*self.ptr.borrow_mut() = Some(self.backend.device.alloc(*self.offset.borrow()));
	}

	pub fn get(&self, name: &str) -> *mut u8 {
		self.get_by_slot(self.slot_id(name))
	}

	pub fn get_by_slot(&self, id: SlotId) -> *mut u8 {
		let ptr = self.ptr.borrow().unwrap();
		let offset = self.slots.borrow()[id.0 as usize].0;
		unsafe { ptr.add(offset) }
	}

	pub fn load_weight(&self, path: &str, skip: impl Fn(&str) -> bool) -> usize {
		let infos = load_safetensors(path).unwrap();
		let bytes = std::fs::read(path).unwrap();
		let mut hit = 0;
		for info in &infos {
			if skip(&info.name) {
				continue;
			}
			let id = *self
				.entries
				.borrow()
				.get(&info.name)
				.unwrap_or_else(|| panic!("文件 tensor {} 在 arena 中不存在", info.name));
			assert_eq!(
				self.shape_by_slot(id),
				info.shape,
				"{} 两边 shape 不一致",
				info.name
			);
			assert_eq!(
				dtype_size(&info.dtype),
				Some(D::SIZE),
				"{} dtype 不匹配",
				info.name
			);
			let src = &bytes[info.start..info.end];
			self.backend.device.copy_from_host_to_device(
				self.get_by_slot(id),
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
