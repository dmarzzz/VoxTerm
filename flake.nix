{
  description = "VoxTerm — local offline voice transcription TUI";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        # Separate pkgs instance with CUDA enabled for llama-cpp GPU support
        pkgsCuda = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        python = pkgs.python312;

        isDarwin = pkgs.stdenv.isDarwin;
        isLinux = pkgs.stdenv.isLinux;

        # Silero VAD — packaged without torch/torchaudio (only the ONNX model
        # file is needed; VoxTerm runs it via onnxruntime directly).
        silero-vad = python.pkgs.buildPythonPackage rec {
          pname = "silero-vad";
          version = "5.1.2";
          pyproject = true;
          src = pkgs.fetchurl {
            url = "https://files.pythonhosted.org/packages/source/s/silero-vad/silero_vad-${version}.tar.gz";
            hash = "sha256-xEKXEWACbS16oK2D8MfuhsiXl6ZSif5iXI6ln8b7go0=";
          };
          build-system = [ python.pkgs.hatchling ];
          dependencies = [ python.pkgs.onnxruntime ];
          # torch/torchaudio are declared deps but VoxTerm only uses the
          # bundled ONNX model file — skip the runtime dep check.
          pythonRemoveDeps = [ "torch" "torchaudio" ];
          # Don't try to import — silero_vad.__init__ imports torch.
          # VoxTerm only uses importlib.util.find_spec to locate the ONNX file.
          pythonImportsCheck = [ ];
        };

        # Runtime Python dependencies
        pythonDeps = ps:
          [
            ps.textual
            ps.sounddevice
            ps.numpy
            ps.onnxruntime
            ps.scipy
            ps.zeroconf
            ps.cryptography
            silero-vad
          ]
          ++ pkgs.lib.optionals isLinux [
            ps.faster-whisper
            ps.torch
            ps.pystray
            ps.pillow
            ps.xlib
          ];

        pythonEnv = python.withPackages pythonDeps;

        # System libraries needed at runtime
        darwinDeps = with pkgs; [
          apple-sdk_15
          swiftPackages.swift
        ];

        linuxDeps = with pkgs; [
          pulseaudio
          alsa-lib
          alsa-plugins
          xdotool
          wtype
          ydotool
          libappindicator-gtk3
          gobject-introspection
        ];

        commonDeps = with pkgs; [
          portaudio
          ffmpeg
        ];

        runtimeDeps =
          commonDeps
          ++ pkgs.lib.optionals isDarwin darwinDeps
          ++ pkgs.lib.optionals isLinux linuxDeps;

        # -- Local LLM server (llama.cpp + pinned model) --

        # Qwen2.5-3B-Instruct Q4_K_M — good balance of size (~2GB) and
        # capability for transcript summarization.
        llmModel = pkgs.fetchurl {
          url = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf";
          hash = "sha256-YmtKZni4ZEIkDjPfgZ4AEy07p93f4c3E+7GOCpYVxi0=";
          name = "qwen2.5-3b-instruct-q4_k_m.gguf";
        };

      in
      {
        packages = {
          default = pkgs.stdenv.mkDerivation {
            pname = "voxterm";
            version = "0.0.0";
            src = pkgs.lib.cleanSource ./.;

            nativeBuildInputs = [ pkgs.makeWrapper ];

            dontBuild = true;

            installPhase = ''
              runHook preInstall

              mkdir -p $out/lib/voxterm
              cp -r \
                audio config.py config_store.py diagnostics.py paths.py \
                tui dictation network summarizer \
                $out/lib/voxterm/

              mkdir -p $out/bin

              makeWrapper ${pythonEnv}/bin/python3 $out/bin/voxterm \
                --prefix PYTHONPATH : "$out/lib/voxterm" \
                --prefix LD_LIBRARY_PATH : "${pkgs.lib.makeLibraryPath runtimeDeps}" \
                --add-flags "-m tui.app"

              makeWrapper ${pythonEnv}/bin/python3 $out/bin/voxterm-dictate \
                --prefix PYTHONPATH : "$out/lib/voxterm" \
                --prefix LD_LIBRARY_PATH : "${pkgs.lib.makeLibraryPath runtimeDeps}" \
                --add-flags "-m dictation"

              runHook postInstall
            '';

            meta = {
              description = "Local offline voice transcription TUI";
              mainProgram = "voxterm";
            };
          };

          # Local LLM server — wraps llama-server with the pinned model.
          # Usage: nix run .#llm-server
          # Exposes OpenAI-compatible API on 127.0.0.1:8081
          llm-server = pkgs.writeShellApplication {
            name = "voxterm-llm-server";
            runtimeInputs = [ pkgsCuda.llama-cpp ];
            text = ''
              port="''${VOXTERM_LLM_PORT:-8081}"
              ctx="''${VOXTERM_LLM_CTX:-8192}"
              echo "Starting local LLM server on 127.0.0.1:$port"
              echo "Model: ${llmModel.name}"
              echo "Context: $ctx tokens"
              exec llama-server \
                --model "${llmModel}" \
                --host 127.0.0.1 \
                --port "$port" \
                --ctx-size "$ctx" \
                --flash-attn auto
            '';
          };

          llm-model = llmModel;
        };

        devShells.default = pkgs.mkShell {
          name = "voxterm";

          packages = with pkgs;
            [
              python
              python.pkgs.pip
              llama-cpp
            ]
            ++ commonDeps
            ++ pkgs.lib.optionals isDarwin darwinDeps
            ++ pkgs.lib.optionals isLinux linuxDeps;

          shellHook = ''
            # Create venv if it doesn't exist
            if [ ! -d .venv ]; then
              echo "Creating Python virtual environment..."
              ${python}/bin/python3 -m venv .venv
            fi
            source .venv/bin/activate

            # Ensure pip can find native libs
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath (commonDeps ++ pkgs.lib.optionals isLinux linuxDeps)}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          ''
            + pkgs.lib.optionalString isDarwin ''
              export DYLD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath commonDeps}''${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
            '';
        };
      }
    );
}
