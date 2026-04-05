# NixOS module for voxterm-llm — sandboxed local LLM inference via llama-swap.
#
# llama-swap manages multiple llama-server instances behind one OpenAI-compatible
# API, hot-swapping models on demand. The kernel enforces that inference never
# reaches the internet: IPAddressDeny=any except localhost.
#
# Usage in your NixOS configuration:
#
#   imports = [ voxterm.nixosModules.llm-server ];
#   services.voxterm-llm.enable = true;
#
# GPU support:
#   services.voxterm-llm.llamaPackage = pkgs.llama-cpp.override { cudaSupport = true; };
#
{ config, lib, pkgs, ... }:

let
  cfg = config.services.voxterm-llm;

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

  defaultModel = pkgs.fetchurl {
    url = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf";
    hash = "sha256-YmtKZni4ZEIkDjPfgZ4AEy07p93f4c3E+7GOCpYVxi0=";
    name = "qwen2.5-3b-instruct-q4_k_m.gguf";
  };

  llamaServer = lib.getExe' cfg.llamaPackage "llama-server";

  # Generate llama-swap config.yaml from Nix options
  swapConfig = pkgs.writeText "llama-swap-config.yaml" (builtins.toJSON {
    healthCheckTimeout = 120;
    logLevel = "info";
    models = lib.mapAttrs (_name: model: {
      cmd = lib.concatStringsSep " " ([
        llamaServer
        "--model" (toString model.model)
        "--port" "\${PORT}"
        "--ctx-size" (toString model.contextSize)
        "--flash-attn" "auto"
      ] ++ model.extraArgs);
    } // lib.optionalAttrs (model.aliases != [ ]) {
      inherit (model) aliases;
    } // lib.optionalAttrs (model.ttl != 0) {
      inherit (model) ttl;
    }) cfg.models;
  });

in
{
  options.services.voxterm-llm = {
    enable = lib.mkEnableOption "VoxTerm local LLM server (sandboxed llama-swap)";

    llamaPackage = lib.mkOption {
      type = lib.types.package;
      default = pkgs.llama-cpp;
      defaultText = lib.literalExpression "pkgs.llama-cpp";
      description = "The llama-cpp package providing llama-server. Override with cudaSupport for GPU.";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 8081;
      description = "Port for the llama-swap API (127.0.0.1 only).";
    };

    models = lib.mkOption {
      type = lib.types.attrsOf (lib.types.submodule {
        options = {
          model = lib.mkOption {
            type = lib.types.path;
            description = "Path to the GGUF model file.";
          };
          contextSize = lib.mkOption {
            type = lib.types.int;
            default = 8192;
            description = "Context window size in tokens.";
          };
          aliases = lib.mkOption {
            type = lib.types.listOf lib.types.str;
            default = [ ];
            description = "Alternative model names for API requests.";
          };
          ttl = lib.mkOption {
            type = lib.types.int;
            default = 0;
            description = "Seconds of idle before auto-unload (0 = never).";
          };
          extraArgs = lib.mkOption {
            type = lib.types.listOf lib.types.str;
            default = [ ];
            description = "Extra arguments passed to llama-server for this model.";
          };
        };
      });
      default = {
        "qwen2.5-3b" = {
          model = defaultModel;
          aliases = [ "summarizer" ];
        };
      };
      description = "Models available for inference. Each spawns a llama-server on demand.";
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.services.voxterm-llm = {
      description = "VoxTerm LLM Server (sandboxed llama-swap)";
      after = [ "network.target" ];
      wantedBy = [ "multi-user.target" ];

      serviceConfig = {
        ExecStart = "${lib.getExe llama-swap} --listen 127.0.0.1:${toString cfg.port} --config ${swapConfig}";

        # -- Identity isolation --
        DynamicUser = true;

        # -- Network sandboxing --
        # Only localhost — neither llama-swap nor its child llama-servers
        # can reach the internet.
        RestrictAddressFamilies = [ "AF_INET" "AF_INET6" "AF_UNIX" ];
        IPAddressAllow = [ "127.0.0.0/8" "::1/128" ];
        IPAddressDeny = "any";

        # -- Filesystem sandboxing --
        ProtectHome = true;
        ProtectSystem = "strict";
        PrivateTmp = true;
        ReadOnlyPaths = [ "/" ];
        NoExecPaths = [ "/" ];
        ExecPaths = [
          "/nix/store"
          "/proc"
        ];

        # -- Privilege sandboxing --
        NoNewPrivileges = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectKernelLogs = true;
        ProtectControlGroups = true;
        ProtectClock = true;
        ProtectHostname = true;
        RestrictRealtime = true;
        RestrictSUIDSGID = true;
        LockPersonality = true;
        MemoryDenyWriteExecute = false; # llama.cpp JIT needs W+X
        SystemCallArchitectures = "native";
        SystemCallFilter = [ "@system-service" "~@privileged" ];
        CapabilityBoundingSet = [ "" ];

        # -- Resource management --
        Restart = "on-failure";
        RestartSec = 5;
      };
    };
  };
}
