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

	/// skip: 跳过; transpose: [out,in]→[in,out] 转置后传到显存;
	/// cpu: 载入 host 内存返回而不进显存(如 embedding 查表)。
	/// 返回值是 cpu 谓词命中的那个 tensor 的原始字节(D 是零大小标签,host 数据一律裸字节)。
	pub fn load_weight(
		&self,
		path: &str,
		skip: impl Fn(&str) -> bool,
		transpose: impl Fn(&str) -> bool,
		cpu: impl Fn(&str) -> bool,
	) -> Option<Vec<u8>> {
		let infos = load_safetensors(path).unwrap();
		let bytes = std::fs::read(path).unwrap();
		let entries = self.entries.borrow();
		let mut cpu_tensor: Option<Vec<u8>> = None;
		for info in &infos {
			if skip(&info.name) {
				continue;
			}
			assert_eq!(
				dtype_size(&info.dtype),
				Some(D::SIZE),
				"{} dtype 不匹配",
				info.name
			);
			let src = &bytes[info.start..info.end];

			if cpu(&info.name) {
				// Option 只有一个坑位; 命中第二个说明谓词写宽了,当场报错
				assert!(
					cpu_tensor.is_none(),
					"cpu 谓词命中了多个 tensor: {}",
					info.name
				);
				cpu_tensor = Some(src.to_vec());
				continue;
			}

			let (_, shape) = entries
				.get(&info.name)
				.unwrap_or_else(|| panic!("文件 tensor {} 在 arena 中不存在", info.name));
			if transpose(&info.name) {
				assert_eq!(info.shape.len(), 2, "{} 只有二维矩阵能转置", info.name);
				assert_eq!(shape, &vec![info.shape[1], info.shape[0]], "{} 转置后 shape 不匹配", info.name);

				let staging = transpose_bytes(src, info.shape[0], info.shape[1], D::SIZE);
				self.backend.device.copy_from_host_to_device(self.get(&info.name), staging.as_ptr(), staging.len());
			} else {
				assert_eq!(shape, &info.shape, "{} 两边 shape 不一致", info.name);
				self.backend.device.copy_from_host_to_device(
					self.get(&info.name),
					src.as_ptr(),
					src.len(),
				);
			}
		}
		cpu_tensor
	}
}

fn transpose_bytes(src: &[u8], rows: usize, cols: usize, es: usize) -> Vec<u8> {
	let mut staging = vec![0u8; src.len()];
	for r in 0..rows {
		for c in 0..cols {
			let s = (r * cols + c) * es;     // 源: (r, c)
			let d = (c * rows + r) * es;     // 目标: (c, r)
			staging[d..d + es].copy_from_slice(&src[s..s + es]);
		}
	}
	staging
}

impl<D: Dtype> Drop for Arena<D> {
	fn drop(&mut self) {
		if let Some(ptr) = *self.ptr.borrow() {
			self.backend.device.free(ptr);
		}
	}
}


#[cfg(test)]
mod tests{
    use crate::arena::transpose_bytes;

    #[test]
	fn test_transpose_bytes() {
		// es=1: [3,2] 的 1 2 / 3 4 / 5 6 转置成 [2,3] 的 1 3 5 / 2 4 6
		let a = vec![1u8, 2, 3, 4, 5, 6];
		assert_eq!(transpose_bytes(&a, 3, 2, 1), vec![1, 3, 5, 2, 4, 6]);
		// es=2: 每个元素两字节,元素内字节序保持
		let a = vec![1u8, 10, 2, 20, 3, 30, 4, 40]; // [2,2] 元素: (1,10) (2,20) / (3,30) (4,40)
		assert_eq!(transpose_bytes(&a, 2, 2, 2), vec![1, 10, 3, 30, 2, 20, 4, 40]);
		// 方阵转置两次 == 原样
		let a: Vec<u8> = (0..36).collect(); // [3,3] es=4
		assert_eq!(transpose_bytes(&transpose_bytes(&a, 3, 3, 4), 3, 3, 4), a);
	}
}
