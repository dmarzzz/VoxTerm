{
  description = "VoxTerm — local offline voice transcription TUI";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    {
      nixosModules.llm-server = ./nix/module.nix;
    }
    //
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

        # -- Local LLM inference (llama-swap + llama-server + pinned model) --

        llama-swap = pkgs.buildGoModule.override { go = pkgs.go_1_26; } {
          pname = "llama-swap";
          version = "199";
          src = pkgs.fetchFromGitHub {
            owner = "mostlygeek";
            repo = "llama-swap";
            rev = "v199";
            hash = "sha256-5dGILqoQWMn+PGxgKdMn3LvWB2U5YrgKy3kE8O+RVeM=";
          };
          vendorHash = "sha256-XiDYlw/byu8CWvg4KSPC7m8PGCZXtp08Y1velx4BR8U=";
          subPackages = [ "." ];
          preBuild = "mkdir -p proxy/ui_dist && touch proxy/ui_dist/placeholder.txt";
          ldflags = [ "-X main.version=199" "-X main.commit=v199" ];
          meta.description = "Hot-swap proxy for local LLM inference servers";
          meta.mainProgram = "llama-swap";
        };

        # Qwen2.5-3B-Instruct Q4_K_M — hash-pinned, auditable provenance.
        llmModel = pkgs.fetchurl {
          url = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf";
          hash = "sha256-YmtKZni4ZEIkDjPfgZ4AEy07p93f4c3E+7GOCpYVxi0=";
          name = "qwen2.5-3b-instruct-q4_k_m.gguf";
        };

        # llama-swap config with the pinned model
        llmConfig = pkgs.writeText "llama-swap-config.yaml" (builtins.toJSON {
          healthCheckTimeout = 120;
          logLevel = "info";
          models = {
            "qwen2.5-3b" = {
              cmd = builtins.concatStringsSep " " [
                "${pkgsCuda.llama-cpp}/bin/llama-server"
                "--model" (toString llmModel)
                "--port" "\${PORT}"
                "--ctx-size" "8192"
                "--flash-attn" "auto"
              ];
              aliases = [ "summarizer" ];
            };
          };
        });

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

          # Local LLM server — llama-swap managing llama-server backends.
          # Usage: nix run .#llm-server
          # Exposes OpenAI-compatible API on 127.0.0.1:8081
          # Models are hot-swapped on demand based on the "model" field in requests.
          llm-server = pkgs.writeShellApplication {
            name = "voxterm-llm-server";
            runtimeInputs = [ llama-swap ];
            text = ''
              port="''${VOXTERM_LLM_PORT:-8081}"
              echo "Starting llama-swap on 127.0.0.1:$port"
              echo "Models: qwen2.5-3b (aliases: summarizer)"
              exec llama-swap --listen "127.0.0.1:$port" --config "${llmConfig}"
            '';
          };

          inherit llama-swap;
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
