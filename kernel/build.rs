fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();

    std::process::Command::new("nvcc")
        .args(&[
            "--compiler-options",
            "-fPIC",
            "-arch=sm_120",
            "-O2",
            "--use_fast_math",
            "-Isrc/CUDA",
            "-c",
            "src/CUDA/kernels.cu",
            "-o",
            &format!("{out_dir}/kernels.o"),
        ])
        .status()
        .unwrap();

    std::process::Command::new("ar")
        .args(&[
            "rcs",
            &format!("{out_dir}/libkernels.a"),
            &format!("{out_dir}/kernels.o"),
        ])
        .status()
        .unwrap();

    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static=kernels");
    println!("cargo:rustc-link-lib=dylib=cudart");
}
