#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
import logging
import argparse
import platform

# Inject env/Scripts for nvinfer.dll on Windows before importing tensorrt
if os.name == 'nt':
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_scripts = os.path.join(app_dir, 'env', 'Scripts')
    if os.path.exists(env_scripts):
        os.environ['PATH'] = env_scripts + os.pathsep + os.environ.get('PATH', '')
        if hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(env_scripts)
            except Exception:
                pass

import tensorrt as trt
import ctypes
import numpy as np

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")
PLUGIN_LOADED = False


def load_plugins(logger: trt.Logger):
    """Load the custom 3D GridSample plugin DLL/SO required by warping_spade."""
    global PLUGIN_LOADED

    # Determine script & checkpoint directories for absolute paths
    script_dir = os.path.dirname(os.path.realpath(__file__))
    app_dir = os.path.dirname(script_dir)
    checkpoint_dir = os.path.join(app_dir, "checkpoints", "liveportrait_onnx")

    if platform.system().lower() == 'linux':
        plugin_path = os.path.join(checkpoint_dir, "libgrid_sample_3d_plugin.so")
        try:
            ctypes.CDLL(plugin_path, mode=ctypes.RTLD_GLOBAL)
            PLUGIN_LOADED = True
        except OSError as e:
            logging.warning(f"Could not load plugin {plugin_path}: {e} — continuing without it")
    else:
        plugin_path = os.path.join(checkpoint_dir, "grid_sample_3d_plugin.dll")

        # On Windows (Python 3.8+), add directories containing the DLL's dependencies
        # to the DLL search path BEFORE loading the plugin.
        env_dir = os.path.join(app_dir, "env")
        site_pkgs = os.path.join(env_dir, "Lib", "site-packages")
        dll_search_dirs = [
            checkpoint_dir,
            os.path.join(site_pkgs, "nvidia", "cudnn", "bin"),
            os.path.join(site_pkgs, "nvidia", "cublas", "bin"),
            os.path.join(site_pkgs, "nvidia", "cuda_runtime", "bin"),
            os.path.join(site_pkgs, "nvidia", "cuda_nvrtc", "bin"),
            os.path.join(site_pkgs, "tensorrt_libs"),
            os.path.join(site_pkgs, "tensorrt"),
        ]
        added = []
        for d in dll_search_dirs:
            if os.path.isdir(d):
                try:
                    added.append(os.add_dll_directory(d))
                except Exception:
                    pass

        try:
            ctypes.CDLL(plugin_path, mode=ctypes.RTLD_GLOBAL, winmode=0)
            logging.info(f"Loaded plugin: {plugin_path}")
            PLUGIN_LOADED = True
        except OSError as e:
            logging.warning(
                f"Could not load {plugin_path}: {e}\n"
                f"NOTE: This plugin was built against TensorRT 8 (nvinfer.dll) and is "
                f"incompatible with TensorRT 10 (nvinfer_10.dll). "
                f"warping_spade will stay on ONNX/CUDA — all other models will use TRT."
            )
        finally:
            for ctx in added:
                try:
                    ctx.close()
                except Exception:
                    pass

    # Initialize TensorRT plugin library
    trt.init_libnvinfer_plugins(logger, "")


class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(self, verbose=False):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        # TensorRT 10: use set_memory_pool_limit instead of max_workspace_size
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 12 * (2 ** 30))  # 12 GB


        self.batch_size = None
        self.network = None
        self.parser = None

        # Load custom plugins
        load_plugins(self.trt_logger)

    def create_network(self, onnx_path):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """
        if hasattr(trt.NetworkDefinitionCreationFlag, 'EXPLICIT_BATCH'):
            network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        else:
            network_flags = 0
        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                log.error("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    log.error(self.parser.get_error(error))
                sys.exit(1)

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        log.info("Network Description")
        for input in inputs:
            self.batch_size = input.shape[0]
            log.info("Input '{}' with shape {} and dtype {}".format(input.name, input.shape, input.dtype))
        for output in outputs:
            log.info("Output '{}' with shape {} and dtype {}".format(output.name, output.shape, output.dtype))

        # Setup optimization profile for dynamic shapes
        profile = self.builder.create_optimization_profile()
        has_dynamic = False
        for i in range(self.network.num_inputs):
            input_tensor = self.network.get_input(i)
            shape = input_tensor.shape
            if any(dim is None or dim <= 0 for dim in shape):
                has_dynamic = True
                min_shape = list(shape)
                opt_shape = list(shape)
                max_shape = list(shape)
                for d in range(len(shape)):
                    val = shape[d]
                    if val is None or val <= 0:
                        if d == 0:
                            min_shape[0] = 1
                            opt_shape[0] = 1
                            max_shape[0] = 4
                        else:
                            min_shape[d] = 112
                            opt_shape[d] = 640
                            max_shape[d] = 2048
                log.info("Setting optimization profile for dynamic input '{}': min={}, opt={}, max={}".format(
                    input_tensor.name, min_shape, opt_shape, max_shape
                ))
                profile.set_shape(input_tensor.name, tuple(min_shape), tuple(opt_shape), tuple(max_shape))
        if has_dynamic:
            self.config.add_optimization_profile(profile)

    def create_engine(
            self,
            engine_path,
            precision
    ):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        log.info("Building {} Engine in {}".format(precision, engine_path))

        if precision == "fp16":
            if not getattr(self.builder, "platform_has_fast_fp16", True):
                log.warning("FP16 is not supported natively on this platform/device")
            else:
                if hasattr(trt.BuilderFlag, "FP16"):
                    self.config.set_flag(trt.BuilderFlag.FP16)
                else:
                    log.warning("TensorRT 11+ does not support trt.BuilderFlag.FP16. Proceeding without it (defaults to ONNX model precision/TF32).")

        log.info("Building serialized engine (this may take several minutes)...")
        serialized_engine = self.builder.build_serialized_network(self.network, self.config)
        if serialized_engine is None:
            log.error("Failed to build TensorRT engine!")
            sys.exit(1)
        with open(engine_path, "wb") as f:
            log.info("Serializing engine to file: {:}".format(engine_path))
            f.write(serialized_engine)


def main(args):
    builder = EngineBuilder(args.verbose)
    builder.create_network(args.onnx)
    builder.create_engine(
        args.engine,
        args.precision
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx", required=True, help="The input ONNX model file to load")
    parser.add_argument("-e", "--engine", help="The output path for the TRT engine")
    parser.add_argument(
        "-p",
        "--precision",
        default="fp16",
        choices=["fp32", "fp16", "int8"],
        help="The precision mode to build in, either 'fp32', 'fp16' or 'int8', default: 'fp16'",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable more verbose log output")
    args = parser.parse_args()
    if args.engine is None:
        args.engine = args.onnx.replace(".onnx", ".trt")
    main(args)
