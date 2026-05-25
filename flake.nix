{
  description = "C++ Torch + OpenCV dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";

      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
        };
      };

      torch = pkgs.libtorch-bin.override {
        cudaSupport = true;
        
      };

      opencv = pkgs.opencv;
      cuda = pkgs.cudaPackages.cudatoolkit;
      cudaLibs = with pkgs.cudaPackages_13; [
        cuda_nvrtc
        cuda_cudart
        libcublas
        libcufft
        libcurand
        libcusolver
        libcusparse
        cudnn
      ];
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          clang-tools
          cmake
          ninja
          pkg-config

          torch
          opencv
          cuda

          # Useful if you compile your own CUDA kernels.
          # cudaPackages.cuda_nvcc
          # cudaPackages.cuda_cudart
        ] ++ cudaLibs;

        CUDA_HOME = "${cuda}";
        CUDA_PATH = "${cuda}";
        CUDA_TOOLKIT_ROOT_DIR = "${cuda}";

        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath([
          torch
          opencv
          pkgs.stdenv.cc.cc.lib
          # pkgs.cudaPackages.cuda_cudart  # Only uncomment this alongside the ones above
        ] ++ cudaLibs);

        CMAKE_PREFIX_PATH = pkgs.lib.makeSearchPathOutput "dev" "lib/cmake" [
          torch
          opencv
        ];

        shellHook = ''
          echo "C++ Torch + OpenCV shell"
          echo "Torch:  ${torch}"
          echo "OpenCV: ${opencv}"
          echo "CUDA:   ${cuda}"
          echo "nvcc:   $(command -v nvcc || true)"
          echo "NVRTC:  ${pkgs.cudaPackages.cuda_nvrtc}"
        '';
      };
    };
}
