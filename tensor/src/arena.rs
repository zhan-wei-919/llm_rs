use backend::{Dtype, Backend};
use std::collections::HashMap;
use std::sync::Arc;
use std::cell::RefCell;

pub struct Arena<D: Dtype> {
    ptr: RefCell<Option<*mut u8>>,
    offset: RefCell<usize>,
    entries: RefCell<HashMap<String, (usize, Vec<usize>)>>,
    pub backend: Arc<Backend<D>>,
}

impl<D: Dtype> Arena<D> {
    pub fn new(backend: Arc<Backend<D>>) -> Self {
        Arena { ptr: RefCell::new(None), offset: RefCell::new(0), entries: RefCell::new(HashMap::new()), backend }
    }

    pub fn alloc(&self, name: String, shape: Vec<usize>) {
    	let numel: usize = shape.iter().product();
    	let size = (numel * D::SIZE + 255) & !255;
    	self.entries.borrow_mut().insert(name, (*self.offset.borrow(), shape));
    	self.offset.borrow_mut() += size
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
}

impl<D: Dtype> Drop for Arena<D> {
    fn drop(&mut self) {
        if let Some(ptr) = *self.ptr.borrow() {self.backend.device.free(ptr);}
    }
}
