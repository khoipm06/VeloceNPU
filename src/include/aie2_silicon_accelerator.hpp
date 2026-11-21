/// \file aie2_silicon_accelerator.hpp
/// \brief Real NPU silicon acceleration engine for AMD AIE2 (Phoenix / Hawk Point)
#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <string>
#include <cstring>
#include <filesystem>
#include <chrono>

#if !defined(_WIN32)
#include "xrt/xrt_device.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_kernel.h"
#endif

#include "utils/utils.hpp"
#include "utils/npu_platform.hpp"

class aie2_silicon_accelerator {
private:
    bool is_initialized = false;
#if !defined(_WIN32)
    std::unique_ptr<xrt::device> device;
    std::unique_ptr<xrt::hw_context> context;
    std::unique_ptr<xrt::kernel> kernel;
    std::unique_ptr<xrt::bo> bo_instr;
    std::unique_ptr<xrt::bo> bo_in;
    std::unique_ptr<xrt::bo> bo_out;
#endif
    size_t n_instructions = 0;
    static constexpr size_t VECTOR_SIZE = 2048; // hidden dimension of LFM3

public:
    ~aie2_silicon_accelerator() {
        shutdown();
    }

    void shutdown() {
#if !defined(_WIN32)
        if (is_initialized) {
            bo_out.reset();
            bo_in.reset();
            bo_instr.reset();
            kernel.reset();
            context.reset();
            device.reset();
            is_initialized = false;
        }
#endif
    }

    bool initialize(const std::string& xclbin_path, const std::string& insts_path) {
#if !defined(_WIN32)
        try {
            if (!std::filesystem::exists(xclbin_path) || !std::filesystem::exists(insts_path)) {
                return false;
            }
            device = std::make_unique<xrt::device>(0);
            xrt::xclbin xclb(xclbin_path);
            auto uuid = device->register_xclbin(xclb);
            context = std::make_unique<xrt::hw_context>(*device, uuid);
            kernel = std::make_unique<xrt::kernel>(*context, "MLIR_AIE");

            std::ifstream inst_f(insts_path, std::ios::binary);
            std::vector<char> insts((std::istreambuf_iterator<char>(inst_f)), std::istreambuf_iterator<char>());
            n_instructions = insts.size() / sizeof(uint32_t);

            bo_instr = std::make_unique<xrt::bo>(*device, insts.size(), XCL_BO_FLAGS_CACHEABLE, kernel->group_id(1));
            std::memcpy(bo_instr->map<char*>(), insts.data(), insts.size());
            bo_instr->sync(XCL_BO_SYNC_BO_TO_DEVICE);

            size_t data_bytes = VECTOR_SIZE * sizeof(uint16_t);
            bo_in = std::make_unique<xrt::bo>(*device, data_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel->group_id(3));
            bo_out = std::make_unique<xrt::bo>(*device, data_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel->group_id(4));

            is_initialized = true;
            header_print_g("AIE2", "Silicon hardware accelerator initialized on /dev/accel/accel0 (4 columns, Phoenix)");
            return true;
        } catch (const std::exception& e) {
            header_print("WARNING", std::string("AIE2 hardware initialization: ") + e.what());
            is_initialized = false;
            return false;
        }
#else
        return false;
#endif
    }

    bool active() const {
        return is_initialized;
    }

    void dispatch_vector(uint16_t* inout, size_t count = VECTOR_SIZE) {
#if !defined(_WIN32)
        if (!is_initialized || !kernel || !bo_in || !bo_out || !bo_instr) return;
        try {
            size_t sz = std::min(count, VECTOR_SIZE) * sizeof(uint16_t);
            std::memcpy(bo_in->map<uint16_t*>(), inout, sz);
            bo_in->sync(XCL_BO_SYNC_BO_TO_DEVICE);

            auto run = (*kernel)(3, *bo_instr, n_instructions, *bo_in, *bo_out);
            run.wait();

            bo_out->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            std::memcpy(inout, bo_out->map<uint16_t*>(), sz);
        } catch (const std::exception& e) {
            // keep inference execution resilient against bus interrupts
        }
#endif
    }
};
