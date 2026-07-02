use std::fmt;
use std::fs;

/*
safetensors 文件布局：

[8 字节] header_size    u64 little-endian，表示后面 JSON header 有多长
[header_size 字节] JSON header
[剩余字节] tensor 数据，连续排列

JSON header 长这样：

{
  "__metadata__": { "format": "pt" },        // 可选，不是 tensor
  "wte.weight": {
    "dtype": "F32",
    "shape": [50257, 768],
    "data_offsets": [0, 154387456]
  },
  ...
}

data_offsets 是 [start, end)，相对于数据段起始位置的字节偏移；
解析时统一换算成文件内的绝对偏移再返回，调用方不需要知道 header 有多长。
*/

#[derive(Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub start: usize,
    pub end: usize,
}

pub fn dtype_size(dtype: &str) -> Option<usize> {
    match dtype {
        "F32" => Some(4),
        "BF16" => Some(2),
        _ => None,
    }
}

#[derive(Debug)]
pub enum SafetensorsError {
    Io(std::io::Error),
    Truncated,
    Header(String),
    Tensor { name: String, reason: String },
}

impl fmt::Display for SafetensorsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SafetensorsError::Io(e) => write!(f, "io error: {}", e),
            SafetensorsError::Truncated => write!(f, "file shorter than header declares"),
            SafetensorsError::Header(reason) => write!(f, "invalid header: {}", reason),
            SafetensorsError::Tensor { name, reason } => {
                write!(f, "invalid tensor {:?}: {}", name, reason)
            }
        }
    }
}

impl std::error::Error for SafetensorsError {}

impl From<std::io::Error> for SafetensorsError {
    fn from(e: std::io::Error) -> Self {
        SafetensorsError::Io(e)
    }
}

pub fn load_safetensors(path: &str) -> Result<Vec<TensorInfo>, SafetensorsError> {
    parse(&fs::read(path)?)
}

fn parse(bytes: &[u8]) -> Result<Vec<TensorInfo>, SafetensorsError> {
    let header_size = bytes.get(..8).ok_or(SafetensorsError::Truncated)?;
    let header_size = u64::from_le_bytes(header_size.try_into().unwrap()) as usize;
    let data_start = 8usize
        .checked_add(header_size)
        .ok_or(SafetensorsError::Truncated)?;
    let header = bytes
        .get(8..data_start)
        .ok_or(SafetensorsError::Truncated)?;

    let header: serde_json::Value = serde_json::from_slice(header)
        .map_err(|e| SafetensorsError::Header(e.to_string()))?;
    let obj = header
        .as_object()
        .ok_or_else(|| SafetensorsError::Header("header is not a JSON object".into()))?;

    let mut model = Vec::new();
    for (name, value) in obj {
        if name == "__metadata__" {
            continue;
        }
        model.push(parse_tensor(name, value, data_start, bytes.len())?);
    }
    Ok(model)
}

fn parse_tensor(
    name: &str,
    value: &serde_json::Value,
    data_start: usize,
    file_len: usize,
) -> Result<TensorInfo, SafetensorsError> {
    let err = |reason: String| SafetensorsError::Tensor { name: name.to_string(), reason };

    let dtype = value["dtype"]
        .as_str()
        .ok_or_else(|| err("missing dtype".into()))?;
    let elem_size =
        dtype_size(dtype).ok_or_else(|| err(format!("unsupported dtype {:?}", dtype)))?;

    let shape = value["shape"]
        .as_array()
        .ok_or_else(|| err("missing shape".into()))?
        .iter()
        .map(|v| v.as_u64().map(|n| n as usize))
        .collect::<Option<Vec<usize>>>()
        .ok_or_else(|| err("shape contains non-integer".into()))?;

    let offsets = value["data_offsets"]
        .as_array()
        .ok_or_else(|| err("missing data_offsets".into()))?;
    let [start, end] = offsets.as_slice() else {
        return Err(err(format!("data_offsets has {} elements, expected 2", offsets.len())));
    };
    let (start, end) = match (start.as_u64(), end.as_u64()) {
        (Some(s), Some(e)) => (s as usize, e as usize),
        _ => return Err(err("data_offsets contains non-integer".into())),
    };

    if start > end || data_start + end > file_len {
        return Err(err(format!(
            "data_offsets [{}, {}) out of range, data section is {} bytes",
            start,
            end,
            file_len - data_start
        )));
    }
    let expected = shape.iter().product::<usize>() * elem_size;
    if end - start != expected {
        return Err(err(format!(
            "data_offsets span {} bytes but shape {:?} x {} needs {}",
            end - start,
            shape,
            dtype,
            expected
        )));
    }

    Ok(TensorInfo {
        name: name.to_string(),
        dtype: dtype.to_string(),
        shape,
        start: data_start + start,
        end: data_start + end,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_load_gpt2() {
        let a: Vec<TensorInfo> = load_safetensors("../weights/model.safetensors").unwrap();
        println!("{:#?}", a);
    }
}