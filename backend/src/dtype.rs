pub trait Dtype {
    const SIZE: usize;
}

pub struct F32;
pub struct BF16;
pub struct F16;

impl Dtype for F32 {
    const SIZE: usize = 4;
}
impl Dtype for BF16 {
    const SIZE: usize = 2;
}
impl Dtype for F16 {
    const SIZE: usize = 2;
}
