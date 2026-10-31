use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cuda/kernel.cu");
    println!("cargo:rerun-if-changed=cuda/grouped_tail.cuh");
    println!("cargo:rerun-if-changed=cuda/compact_chain.cuh");
    println!("cargo:rerun-if-changed=cuda/xtail.cuh");
    println!("cargo:rerun-if-changed=cuda/screen.cuh");
    println!("cargo:rerun-if-changed=cuda/jdip.cuh");
    println!("cargo:rerun-if-changed=cuda/jdscreen.cuh");
    println!("cargo:rerun-if-changed=cuda/wq_jump.cuh");
    println!("cargo:rerun-if-changed=cuda/wq_jd.cuh");
    println!("cargo:rerun-if-changed=cuda/twophase.cuh");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let cuda_home = env::var_os("CUDA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
    let nvcc = cuda_home.join("bin/nvcc");
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "native".into());
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let obj = out.join("kernel.o");
    let lib = out.join("libcvkernel.a");
    let status = Command::new(&nvcc)
        .args(["-O3", "-std=c++17", &format!("-arch={arch}"), "-Xcompiler", "-fPIC", "-c", "cuda/kernel.cu", "-o"])
        .arg(&obj)
        .status()
        .expect("failed to run nvcc (set CUDA_HOME)");
    assert!(status.success(), "nvcc failed");
    let status = Command::new("ar").arg("crs").arg(&lib).arg(&obj).status().expect("ar");
    assert!(status.success(), "ar failed");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=cvkernel");
    println!("cargo:rustc-link-search=native={}", cuda_home.join("lib64").display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
