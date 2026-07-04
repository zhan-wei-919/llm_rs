pub trait Dtype: Copy + Clone {
	const SIZE: usize;
	// dtype 分发标签,与 C++ kernel 侧 switch(dtype) 的 case 一一对应,是跨 FFI 的唯一契约
	const TAG: i32;
}

#[derive(Clone, Copy)]
pub struct F32;
#[derive(Clone, Copy)]
pub struct BF16;
#[derive(Clone, Copy)]
pub struct F16;

impl Dtype for F32 {
	const SIZE: usize = 4;
	const TAG: i32 = 0;
}
impl Dtype for BF16 {
	const SIZE: usize = 2;
	const TAG: i32 = 1;
}
impl Dtype for F16 {
	const SIZE: usize = 2;
	const TAG: i32 = 2;
}
