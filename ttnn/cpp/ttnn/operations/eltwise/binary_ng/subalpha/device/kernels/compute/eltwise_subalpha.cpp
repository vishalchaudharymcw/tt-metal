// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/negative.h"
#include "debug/dprint.h"

namespace NAMESPACE {

ALWI void process_tile(
    tt::CBIndex cb_pre_lhs, // a
    tt::CBIndex cb_alpha, // alpha
    tt::CBIndex cb_pre_rhs,  // b
    tt::CBIndex cb_post_rhs, // cb_inter
    tt::CBIndex cb_out,
    uint32_t freq,
    uint32_t tile_start,
    uint32_t num_tiles_per_cycle) {
    using namespace ckernel;

    #if BCAST_INPUT
    #define CB_PRE_BCAST cb_pre_rhs
    #define CB_PRE_OTHER cb_pre_lhs
    #else
    #define CB_PRE_BCAST cb_pre_lhs
    #define CB_PRE_OTHER cb_pre_rhs
    #endif

    // B is broadcasted
    if(BCAST_INPUT) {

        cb_wait_front(CB_PRE_BCAST, num_tiles_per_cycle); // wait for b
        cb_reserve_back(cb_post_rhs, num_tiles_per_cycle);   // reserve cb_inter

        mul_tiles_init(CB_PRE_BCAST, cb_alpha);
        tile_regs_acquire();
        mul_tiles(CB_PRE_BCAST, cb_alpha, 0, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_post_rhs);
        tile_regs_release();

        cb_push_back(cb_post_rhs, num_tiles_per_cycle);
        cb_wait_front(cb_post_rhs, num_tiles_per_cycle); //cb_inter

        // Compute on other cb (no-bcast reqd) input_a
        for (uint32_t j = tile_start; j < freq; ++j) {

            cb_wait_front(CB_PRE_OTHER, num_tiles_per_cycle); // input_a
            cb_reserve_back(cb_out, num_tiles_per_cycle);

            sub_tiles_init(CB_PRE_OTHER, cb_post_rhs);
            tile_regs_acquire();
            sub_tiles(CB_PRE_OTHER, cb_post_rhs, 0, 0, 0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, cb_out);

            tile_regs_release();

            cb_pop_front(CB_PRE_OTHER, num_tiles_per_cycle); // pop-a
            cb_push_back(cb_out, num_tiles_per_cycle);
        }
        cb_pop_front(CB_PRE_BCAST, num_tiles_per_cycle); // pop-b
        cb_pop_front(cb_post_rhs, num_tiles_per_cycle); // cb_inter

    }
    // A is broadcasted
    else{
        cb_wait_front(CB_PRE_BCAST, num_tiles_per_cycle); // wait for a

        for (uint32_t j = tile_start; j < freq; ++j) {

            cb_wait_front(cb_pre_rhs, num_tiles_per_cycle); // input_b
            cb_reserve_back(cb_post_rhs, num_tiles_per_cycle); //cb_inter

            mul_tiles_init(CB_PRE_OTHER, cb_alpha);
            tile_regs_acquire();
            mul_tiles(CB_PRE_OTHER, cb_alpha, 0, 0, 0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, cb_post_rhs);

            tile_regs_release();

            cb_pop_front(CB_PRE_OTHER, num_tiles_per_cycle); //pop b

            cb_push_back(cb_post_rhs, num_tiles_per_cycle);
            cb_wait_front(cb_post_rhs, num_tiles_per_cycle);
            cb_reserve_back(cb_out, num_tiles_per_cycle);   // reserve cb_out

            sub_tiles_init(CB_PRE_BCAST, cb_post_rhs);

            tile_regs_acquire();
            sub_tiles(CB_PRE_BCAST, cb_post_rhs, 0, 0, 0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();

            cb_push_back(cb_out, num_tiles_per_cycle);

            cb_pop_front(cb_post_rhs, num_tiles_per_cycle);
        }

        cb_pop_front(CB_PRE_BCAST, num_tiles_per_cycle); // pop-a
    }
}

void MAIN {

    using namespace ckernel;
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_input_a = tt::CBIndex::c_0;
    constexpr auto cb_input_b = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    constexpr auto cb_alpha = tt::CBIndex::c_3;  // alpha
    constexpr auto cb_inter = tt::CBIndex::c_4;  // intermediate cb

    binary_op_init_common(cb_input_a, cb_inter, cb_out);

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    cb_wait_front(cb_alpha, num_tiles_per_cycle);

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile(cb_input_a, cb_alpha, cb_input_b, cb_inter, cb_out, tile_freq, tile_start, num_tiles_per_cycle);
    }

    if (remaining_iterations > 0) {
        process_tile(cb_input_a, cb_alpha, cb_input_b, cb_inter, cb_out, remaining_iterations, tile_start, num_tiles_per_cycle);
    }
}

}  // namespace NAMESPACE
