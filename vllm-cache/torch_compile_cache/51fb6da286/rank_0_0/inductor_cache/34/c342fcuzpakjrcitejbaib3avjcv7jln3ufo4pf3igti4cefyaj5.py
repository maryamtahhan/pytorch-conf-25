# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /home/vllm/.cache/vllm/torch_compile_cache/51fb6da286/rank_0_0/inductor_cache/k6/ck67xyn7nz4hq2ybqarel4ww7yo7ksdmqk7gyof3m7e6pl55yeju.py
# Topologically Sorted Source Nodes: [long, embedding], Original ATen: [aten._to_copy, aten.embedding]
# Source node to ATen node mapping:
#   embedding => embedding
#   long => convert_element_type
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.int64), kwargs = {})
#   %embedding : [num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %convert_element_type), kwargs = {})
triton_poi_fused__to_copy_embedding_0 = async_compile.triton('triton_poi_fused__to_copy_embedding_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=104, cc='gfx90a', major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_embedding_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4ABBB60B77990E471E335842B31931E2FC598AFF563B388D353CC582A1921888', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'is_hip': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_embedding_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = xindex // 4096
    x0 = (xindex % 4096)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tl.full([XBLOCK], 128256, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert((0 <= tmp5) & (tmp5 < 128256), "index out of bounds: 0 <= tmp5 < 128256")
    tmp7 = tl.load(in_ptr1 + (x0 + 4096*tmp5), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp7, None)
''', device_str='cuda')


# kernel path: /home/vllm/.cache/vllm/torch_compile_cache/51fb6da286/rank_0_0/inductor_cache/2h/c2h2po2sjromnyxfdwdgfyrcjxoujqkeyrh3bo7nvthtmpqcydsh.py
# Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_19, %add_75], -1), kwargs = {})
triton_poi_fused_cat_1 = async_compile.triton('triton_poi_fused_cat_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i64', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=104, cc='gfx90a', major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '4ABBB60B77990E471E335842B31931E2FC598AFF563B388D353CC582A1921888', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'is_hip': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 32)
    x2 = xindex // 4096
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp4.to(tl.int1)
    tmp6 = tl.load(in_ptr0 + (128*x1 + 6144*x2 + (x0)), tmp5, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (x2), tmp5, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.full([XBLOCK], 131072, tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert(((0 <= tl.broadcast_to(tmp11, [XBLOCK])) & (tl.broadcast_to(tmp11, [XBLOCK]) < 131072)) | ~(tmp5), "index out of bounds: 0 <= tl.broadcast_to(tmp11, [XBLOCK]) < 131072")
    tmp13 = tl.load(in_ptr2 + (128*tmp11 + (x0)), tmp5, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp14 = tmp6 * tmp13
    tmp15 = tl.load(in_ptr0 + (64 + 128*x1 + 6144*x2 + (x0)), tmp5, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp16 = tl.load(in_ptr2 + (64 + 128*tmp11 + (x0)), tmp5, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp14 - tmp17
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp5, tmp18, tmp19)
    tmp21 = tmp0 >= tmp3
    tmp22 = tl.full([1], 128, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21.to(tl.int1)
    tmp25 = tl.load(in_ptr0 + (64 + 128*x1 + 6144*x2 + ((-64) + x0)), tmp24, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = tl.load(in_ptr1 + (x2), tmp24, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.full([XBLOCK], 131072, tl.int32)
    tmp28 = tmp26 + tmp27
    tmp29 = tmp26 < 0
    tmp30 = tl.where(tmp29, tmp28, tmp26)
    tl.device_assert(((0 <= tl.broadcast_to(tmp30, [XBLOCK])) & (tl.broadcast_to(tmp30, [XBLOCK]) < 131072)) | ~(tmp24), "index out of bounds: 0 <= tl.broadcast_to(tmp30, [XBLOCK]) < 131072")
    tmp32 = tl.load(in_ptr2 + (128*tmp30 + ((-64) + x0)), tmp24, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp33 = tmp25 * tmp32
    tmp34 = tl.load(in_ptr0 + (128*x1 + 6144*x2 + ((-64) + x0)), tmp24, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp35 = tl.load(in_ptr2 + (64 + 128*tmp30 + ((-64) + x0)), tmp24, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tmp34 * tmp35
    tmp37 = tmp33 + tmp36
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp24, tmp37, tmp38)
    tmp40 = tl.where(tmp4, tmp20, tmp39)
    tl.store(out_ptr0 + (x4), tmp40, None)
''', device_str='cuda')


# kernel path: /home/vllm/.cache/vllm/torch_compile_cache/51fb6da286/rank_0_0/inductor_cache/zf/czfblgicyct7dbaobrfr466qlzquxd4roqhj7g3nsfc2tcfmvhyi.py
# Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_2 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_36, %add_137], -1), kwargs = {})
triton_poi_fused_cat_2 = async_compile.triton('triton_poi_fused_cat_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i64', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=104, cc='gfx90a', major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '4ABBB60B77990E471E335842B31931E2FC598AFF563B388D353CC582A1921888', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'is_hip': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 8)
    x2 = xindex // 1024
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp4.to(tl.int1)
    tmp6 = tl.load(in_ptr0 + (4096 + 128*x1 + 6144*x2 + (x0)), xmask & tmp5, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (x2), xmask & tmp5, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.full([XBLOCK], 131072, tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert(((0 <= tl.broadcast_to(tmp11, [XBLOCK])) & (tl.broadcast_to(tmp11, [XBLOCK]) < 131072)) | ~(xmask & tmp5), "index out of bounds: 0 <= tl.broadcast_to(tmp11, [XBLOCK]) < 131072")
    tmp13 = tl.load(in_ptr2 + (128*tmp11 + (x0)), xmask & tmp5, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp14 = tmp6 * tmp13
    tmp15 = tl.load(in_ptr0 + (4160 + 128*x1 + 6144*x2 + (x0)), xmask & tmp5, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp16 = tl.load(in_ptr2 + (64 + 128*tmp11 + (x0)), xmask & tmp5, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp14 - tmp17
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp5, tmp18, tmp19)
    tmp21 = tmp0 >= tmp3
    tmp22 = tl.full([1], 128, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21.to(tl.int1)
    tmp25 = tl.load(in_ptr0 + (4160 + 128*x1 + 6144*x2 + ((-64) + x0)), xmask & tmp24, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = tl.load(in_ptr1 + (x2), xmask & tmp24, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.full([XBLOCK], 131072, tl.int32)
    tmp28 = tmp26 + tmp27
    tmp29 = tmp26 < 0
    tmp30 = tl.where(tmp29, tmp28, tmp26)
    tl.device_assert(((0 <= tl.broadcast_to(tmp30, [XBLOCK])) & (tl.broadcast_to(tmp30, [XBLOCK]) < 131072)) | ~(xmask & tmp24), "index out of bounds: 0 <= tl.broadcast_to(tmp30, [XBLOCK]) < 131072")
    tmp32 = tl.load(in_ptr2 + (128*tmp30 + ((-64) + x0)), xmask & tmp24, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp33 = tmp25 * tmp32
    tmp34 = tl.load(in_ptr0 + (4096 + 128*x1 + 6144*x2 + ((-64) + x0)), xmask & tmp24, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp35 = tl.load(in_ptr2 + (64 + 128*tmp30 + ((-64) + x0)), xmask & tmp24, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tmp34 * tmp35
    tmp37 = tmp33 + tmp36
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp24, tmp37, tmp38)
    tmp40 = tl.where(tmp4, tmp20, tmp39)
    tl.store(out_ptr0 + (x4), tmp40, xmask)
''', device_str='cuda')


# kernel path: /home/vllm/.cache/vllm/torch_compile_cache/51fb6da286/rank_0_0/inductor_cache/pf/cpfcjzg473xr4ozxzbbaoqtikiai3vg4oo5jw2fdxusn4hxmpfib.py
# Topologically Sorted Source Nodes: [view_3], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   view_3 => full_default
# Graph fragment:
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([%arg1_1, 32, 128], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_view_3 = async_compile.triton('triton_poi_fused_view_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=104, cc='gfx90a', major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '4ABBB60B77990E471E335842B31931E2FC598AFF563B388D353CC582A1921888', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'is_hip': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_view_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1 = args
    args.clear()
    s0 = arg1_1
    assert_size_stride(arg0_1, (s0, ), (1, ))
    assert_size_stride(arg2_1, (128256, 4096), (4096, 1))
    assert_size_stride(arg3_1, (4096, ), (1, ))
    assert_size_stride(arg4_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg5_1, (s0, ), (1, ))
    assert_size_stride(arg6_1, (131072, 128), (128, 1))
    assert_size_stride(arg7_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg8_1, (4096, ), (1, ))
    assert_size_stride(arg9_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg10_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg11_1, (4096, ), (1, ))
    assert_size_stride(arg12_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg13_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg14_1, (4096, ), (1, ))
    assert_size_stride(arg15_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg16_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg17_1, (4096, ), (1, ))
    assert_size_stride(arg18_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg19_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg20_1, (4096, ), (1, ))
    assert_size_stride(arg21_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg22_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg23_1, (4096, ), (1, ))
    assert_size_stride(arg24_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg25_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg26_1, (4096, ), (1, ))
    assert_size_stride(arg27_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg28_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg29_1, (4096, ), (1, ))
    assert_size_stride(arg30_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg31_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg32_1, (4096, ), (1, ))
    assert_size_stride(arg33_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg34_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg35_1, (4096, ), (1, ))
    assert_size_stride(arg36_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg37_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg38_1, (4096, ), (1, ))
    assert_size_stride(arg39_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg40_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg41_1, (4096, ), (1, ))
    assert_size_stride(arg42_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg43_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg44_1, (4096, ), (1, ))
    assert_size_stride(arg45_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg46_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg47_1, (4096, ), (1, ))
    assert_size_stride(arg48_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg49_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg50_1, (4096, ), (1, ))
    assert_size_stride(arg51_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg52_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg53_1, (4096, ), (1, ))
    assert_size_stride(arg54_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg55_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg56_1, (4096, ), (1, ))
    assert_size_stride(arg57_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg58_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg59_1, (4096, ), (1, ))
    assert_size_stride(arg60_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg61_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg62_1, (4096, ), (1, ))
    assert_size_stride(arg63_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg64_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg65_1, (4096, ), (1, ))
    assert_size_stride(arg66_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg67_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg68_1, (4096, ), (1, ))
    assert_size_stride(arg69_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg70_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg71_1, (4096, ), (1, ))
    assert_size_stride(arg72_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg73_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg74_1, (4096, ), (1, ))
    assert_size_stride(arg75_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg76_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg77_1, (4096, ), (1, ))
    assert_size_stride(arg78_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg79_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg80_1, (4096, ), (1, ))
    assert_size_stride(arg81_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg82_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg83_1, (4096, ), (1, ))
    assert_size_stride(arg84_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg85_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg86_1, (4096, ), (1, ))
    assert_size_stride(arg87_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg88_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg89_1, (4096, ), (1, ))
    assert_size_stride(arg90_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg91_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg92_1, (4096, ), (1, ))
    assert_size_stride(arg93_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg94_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg95_1, (4096, ), (1, ))
    assert_size_stride(arg96_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg97_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg98_1, (4096, ), (1, ))
    assert_size_stride(arg99_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg100_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg101_1, (4096, ), (1, ))
    assert_size_stride(arg102_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg103_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg104_1, (4096, ), (1, ))
    assert_size_stride(arg105_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg106_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg107_1, (4096, ), (1, ))
    assert_size_stride(arg108_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg109_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg110_1, (4096, ), (1, ))
    assert_size_stride(arg111_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg112_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg113_1, (4096, ), (1, ))
    assert_size_stride(arg114_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg115_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg116_1, (4096, ), (1, ))
    assert_size_stride(arg117_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg118_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg119_1, (4096, ), (1, ))
    assert_size_stride(arg120_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg121_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg122_1, (4096, ), (1, ))
    assert_size_stride(arg123_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg124_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg125_1, (4096, ), (1, ))
    assert_size_stride(arg126_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg127_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg128_1, (4096, ), (1, ))
    assert_size_stride(arg129_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg130_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg131_1, (4096, ), (1, ))
    assert_size_stride(arg132_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg133_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg134_1, (4096, ), (1, ))
    assert_size_stride(arg135_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg136_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg137_1, (4096, ), (1, ))
    assert_size_stride(arg138_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg139_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg140_1, (4096, ), (1, ))
    assert_size_stride(arg141_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg142_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg143_1, (4096, ), (1, ))
    assert_size_stride(arg144_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg145_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg146_1, (4096, ), (1, ))
    assert_size_stride(arg147_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg148_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg149_1, (4096, ), (1, ))
    assert_size_stride(arg150_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg151_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg152_1, (4096, ), (1, ))
    assert_size_stride(arg153_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg154_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg155_1, (4096, ), (1, ))
    assert_size_stride(arg156_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg157_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg158_1, (4096, ), (1, ))
    assert_size_stride(arg159_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg160_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg161_1, (4096, ), (1, ))
    assert_size_stride(arg162_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg163_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg164_1, (4096, ), (1, ))
    assert_size_stride(arg165_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg166_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg167_1, (4096, ), (1, ))
    assert_size_stride(arg168_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg169_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg170_1, (4096, ), (1, ))
    assert_size_stride(arg171_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg172_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg173_1, (4096, ), (1, ))
    assert_size_stride(arg174_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg175_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg176_1, (4096, ), (1, ))
    assert_size_stride(arg177_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg178_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg179_1, (4096, ), (1, ))
    assert_size_stride(arg180_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg181_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg182_1, (4096, ), (1, ))
    assert_size_stride(arg183_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg184_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg185_1, (4096, ), (1, ))
    assert_size_stride(arg186_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg187_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg188_1, (4096, ), (1, ))
    assert_size_stride(arg189_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg190_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg191_1, (4096, ), (1, ))
    assert_size_stride(arg192_1, (6144, 4096), (4096, 1))
    assert_size_stride(arg193_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg194_1, (4096, ), (1, ))
    assert_size_stride(arg195_1, (28672, 4096), (4096, 1))
    assert_size_stride(arg196_1, (4096, 14336), (14336, 1))
    assert_size_stride(arg197_1, (4096, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((s0, 4096), (4096, 1), torch.bfloat16)
        buf1 = empty_strided_cuda((s0, 4096), (4096, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [long, embedding], Original ATen: [aten._to_copy, aten.embedding]
        triton_poi_fused__to_copy_embedding_0_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_embedding_0.run(arg0_1, arg2_1, buf1, triton_poi_fused__to_copy_embedding_0_xnumel, stream=stream0)
        del arg0_1
        del arg2_1
        # Topologically Sorted Source Nodes: [long, embedding], Original ATen: [aten._to_copy, aten.embedding]
        torch.ops._C.rms_norm.default(result=buf0, input=buf1, weight=arg3_1, epsilon=1e-05)
        del arg3_1
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf4 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf0, arg4_1)
        del arg4_1
        buf5 = buf4
        assert_size_stride(buf5, (s0, 6144), (6144, 1))
        del buf4
        buf6 = reinterpret_tensor(buf0, (s0, 32, 128), (4096, 128, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf5, arg5_1, arg6_1, buf6, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf7 = empty_strided_cuda((s0, 8, 128), (1024, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf5, arg5_1, arg6_1, buf7, triton_poi_fused_cat_2_xnumel, stream=stream0)
        buf8 = empty_strided_cuda((s0, 32, 128), (4096, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [view_3], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf8, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_3, unified_attention_with_output], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf6, buf7, reinterpret_tensor(buf5, (s0, 8, 128), (6144, 128, 1), 5120), buf8, 'model.layers.0.self_attn.attn')
        del buf5
        buf11 = reinterpret_tensor(buf6, (s0, 4096), (4096, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_1], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf12 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf8, (s0, 4096), (4096, 1), 0), arg7_1)
        del arg7_1
        buf13 = buf12
        assert_size_stride(buf13, (s0, 4096), (4096, 1))
        del buf12
        buf14 = reinterpret_tensor(buf8, (s0, 4096), (4096, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf11, input=buf13, residual_out=buf14, residual=buf1, weight=arg8_1, epsilon=1e-05)
        del arg8_1
        del buf1
        buf18 = empty_strided_cuda((s0, 14336), (14336, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_2], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf19 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf11, arg9_1)
        del arg9_1
        buf20 = buf19
        assert_size_stride(buf20, (s0, 28672), (28672, 1))
        del buf19
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf18, buf20)
        del buf20
        buf23 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_3], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf24 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf18, arg10_1)
        del arg10_1
        buf25 = buf24
        assert_size_stride(buf25, (s0, 4096), (4096, 1))
        del buf24
        buf26 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf23, input=buf25, residual_out=buf26, residual=buf14, weight=arg11_1, epsilon=1e-05)
        del arg11_1
        del buf14
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_4], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf30 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf23, arg12_1)
        del arg12_1
        buf31 = buf30
        assert_size_stride(buf31, (s0, 6144), (6144, 1))
        del buf30
        buf32 = reinterpret_tensor(buf23, (s0, 32, 128), (4096, 128, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [cat_4], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf31, arg5_1, arg6_1, buf32, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf33 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [cat_6], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf31, arg5_1, arg6_1, buf33, triton_poi_fused_cat_2_xnumel, stream=stream0)
        buf34 = reinterpret_tensor(buf25, (s0, 32, 128), (4096, 128, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [view_10], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf34, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_10, unified_attention_with_output_1], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf32, buf33, reinterpret_tensor(buf31, (s0, 8, 128), (6144, 128, 1), 5120), buf34, 'model.layers.1.self_attn.attn')
        del buf31
        buf37 = reinterpret_tensor(buf32, (s0, 4096), (4096, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_5], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf38 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf34, (s0, 4096), (4096, 1), 0), arg13_1)
        del arg13_1
        buf39 = buf38
        assert_size_stride(buf39, (s0, 4096), (4096, 1))
        del buf38
        buf40 = reinterpret_tensor(buf34, (s0, 4096), (4096, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf37, input=buf39, residual_out=buf40, residual=buf26, weight=arg14_1, epsilon=1e-05)
        del arg14_1
        del buf26
        buf44 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_6], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf45 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf37, arg15_1)
        del arg15_1
        buf46 = buf45
        assert_size_stride(buf46, (s0, 28672), (28672, 1))
        del buf45
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf44, buf46)
        del buf46
        buf49 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_7], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf50 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf44, arg16_1)
        del arg16_1
        buf51 = buf50
        assert_size_stride(buf51, (s0, 4096), (4096, 1))
        del buf50
        buf52 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf49, input=buf51, residual_out=buf52, residual=buf40, weight=arg17_1, epsilon=1e-05)
        del arg17_1
        del buf40
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_8], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf56 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf49, arg18_1)
        del arg18_1
        buf57 = buf56
        assert_size_stride(buf57, (s0, 6144), (6144, 1))
        del buf56
        buf58 = reinterpret_tensor(buf49, (s0, 32, 128), (4096, 128, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [cat_8], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf57, arg5_1, arg6_1, buf58, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf59 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [cat_10], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf57, arg5_1, arg6_1, buf59, triton_poi_fused_cat_2_xnumel, stream=stream0)
        buf60 = reinterpret_tensor(buf51, (s0, 32, 128), (4096, 128, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [view_17], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf60, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_17, unified_attention_with_output_2], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf58, buf59, reinterpret_tensor(buf57, (s0, 8, 128), (6144, 128, 1), 5120), buf60, 'model.layers.2.self_attn.attn')
        del buf57
        buf63 = reinterpret_tensor(buf58, (s0, 4096), (4096, 1), 0); del buf58  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_9], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf64 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf60, (s0, 4096), (4096, 1), 0), arg19_1)
        del arg19_1
        buf65 = buf64
        assert_size_stride(buf65, (s0, 4096), (4096, 1))
        del buf64
        buf66 = reinterpret_tensor(buf60, (s0, 4096), (4096, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf63, input=buf65, residual_out=buf66, residual=buf52, weight=arg20_1, epsilon=1e-05)
        del arg20_1
        del buf52
        buf70 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_10], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf71 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf63, arg21_1)
        del arg21_1
        buf72 = buf71
        assert_size_stride(buf72, (s0, 28672), (28672, 1))
        del buf71
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf70, buf72)
        del buf72
        buf75 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_11], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf76 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf70, arg22_1)
        del arg22_1
        buf77 = buf76
        assert_size_stride(buf77, (s0, 4096), (4096, 1))
        del buf76
        buf78 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf75, input=buf77, residual_out=buf78, residual=buf66, weight=arg23_1, epsilon=1e-05)
        del arg23_1
        del buf66
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_12], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf82 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf75, arg24_1)
        del arg24_1
        buf83 = buf82
        assert_size_stride(buf83, (s0, 6144), (6144, 1))
        del buf82
        buf84 = reinterpret_tensor(buf75, (s0, 32, 128), (4096, 128, 1), 0); del buf75  # reuse
        # Topologically Sorted Source Nodes: [cat_12], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf83, arg5_1, arg6_1, buf84, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf85 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [cat_14], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf83, arg5_1, arg6_1, buf85, triton_poi_fused_cat_2_xnumel, stream=stream0)
        buf86 = reinterpret_tensor(buf77, (s0, 32, 128), (4096, 128, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [view_24], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf86, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_24, unified_attention_with_output_3], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf84, buf85, reinterpret_tensor(buf83, (s0, 8, 128), (6144, 128, 1), 5120), buf86, 'model.layers.3.self_attn.attn')
        del buf83
        buf89 = reinterpret_tensor(buf84, (s0, 4096), (4096, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_13], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf90 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf86, (s0, 4096), (4096, 1), 0), arg25_1)
        del arg25_1
        buf91 = buf90
        assert_size_stride(buf91, (s0, 4096), (4096, 1))
        del buf90
        buf92 = reinterpret_tensor(buf86, (s0, 4096), (4096, 1), 0); del buf86  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf89, input=buf91, residual_out=buf92, residual=buf78, weight=arg26_1, epsilon=1e-05)
        del arg26_1
        del buf78
        buf96 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_14], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf97 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf89, arg27_1)
        del arg27_1
        buf98 = buf97
        assert_size_stride(buf98, (s0, 28672), (28672, 1))
        del buf97
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf96, buf98)
        del buf98
        buf101 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_15], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf102 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf96, arg28_1)
        del arg28_1
        buf103 = buf102
        assert_size_stride(buf103, (s0, 4096), (4096, 1))
        del buf102
        buf104 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf101, input=buf103, residual_out=buf104, residual=buf92, weight=arg29_1, epsilon=1e-05)
        del arg29_1
        del buf103
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_16], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf108 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf101, arg30_1)
        del arg30_1
        buf109 = buf108
        assert_size_stride(buf109, (s0, 6144), (6144, 1))
        del buf108
        buf110 = reinterpret_tensor(buf101, (s0, 32, 128), (4096, 128, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [cat_16], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf109, arg5_1, arg6_1, buf110, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf111 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [cat_18], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf109, arg5_1, arg6_1, buf111, triton_poi_fused_cat_2_xnumel, stream=stream0)
        buf112 = reinterpret_tensor(buf92, (s0, 32, 128), (4096, 128, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [view_31], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf112, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_31, unified_attention_with_output_4], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf110, buf111, reinterpret_tensor(buf109, (s0, 8, 128), (6144, 128, 1), 5120), buf112, 'model.layers.4.self_attn.attn')
        del buf109
        buf115 = reinterpret_tensor(buf110, (s0, 4096), (4096, 1), 0); del buf110  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_17], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf116 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf112, (s0, 4096), (4096, 1), 0), arg31_1)
        del arg31_1
        buf117 = buf116
        assert_size_stride(buf117, (s0, 4096), (4096, 1))
        del buf116
        buf118 = reinterpret_tensor(buf112, (s0, 4096), (4096, 1), 0); del buf112  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf115, input=buf117, residual_out=buf118, residual=buf104, weight=arg32_1, epsilon=1e-05)
        del arg32_1
        del buf104
        buf122 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_18], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf123 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf115, arg33_1)
        del arg33_1
        buf124 = buf123
        assert_size_stride(buf124, (s0, 28672), (28672, 1))
        del buf123
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf122, buf124)
        del buf124
        buf127 = buf115; del buf115  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_19], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf128 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf122, arg34_1)
        del arg34_1
        buf129 = buf128
        assert_size_stride(buf129, (s0, 4096), (4096, 1))
        del buf128
        buf130 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf127, input=buf129, residual_out=buf130, residual=buf118, weight=arg35_1, epsilon=1e-05)
        del arg35_1
        del buf118
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_20], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf134 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf127, arg36_1)
        del arg36_1
        buf135 = buf134
        assert_size_stride(buf135, (s0, 6144), (6144, 1))
        del buf134
        buf136 = reinterpret_tensor(buf127, (s0, 32, 128), (4096, 128, 1), 0); del buf127  # reuse
        # Topologically Sorted Source Nodes: [cat_20], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf135, arg5_1, arg6_1, buf136, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf137 = buf111; del buf111  # reuse
        # Topologically Sorted Source Nodes: [cat_22], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf135, arg5_1, arg6_1, buf137, triton_poi_fused_cat_2_xnumel, stream=stream0)
        buf138 = reinterpret_tensor(buf129, (s0, 32, 128), (4096, 128, 1), 0); del buf129  # reuse
        # Topologically Sorted Source Nodes: [view_38], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf138, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_38, unified_attention_with_output_5], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf136, buf137, reinterpret_tensor(buf135, (s0, 8, 128), (6144, 128, 1), 5120), buf138, 'model.layers.5.self_attn.attn')
        del buf135
        buf141 = reinterpret_tensor(buf136, (s0, 4096), (4096, 1), 0); del buf136  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_21], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf142 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf138, (s0, 4096), (4096, 1), 0), arg37_1)
        del arg37_1
        buf143 = buf142
        assert_size_stride(buf143, (s0, 4096), (4096, 1))
        del buf142
        buf144 = reinterpret_tensor(buf138, (s0, 4096), (4096, 1), 0); del buf138  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf141, input=buf143, residual_out=buf144, residual=buf130, weight=arg38_1, epsilon=1e-05)
        del arg38_1
        del buf130
        buf148 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_22], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf149 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf141, arg39_1)
        del arg39_1
        buf150 = buf149
        assert_size_stride(buf150, (s0, 28672), (28672, 1))
        del buf149
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf148, buf150)
        del buf150
        buf153 = buf141; del buf141  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_23], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf154 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf148, arg40_1)
        del arg40_1
        buf155 = buf154
        assert_size_stride(buf155, (s0, 4096), (4096, 1))
        del buf154
        buf156 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf153, input=buf155, residual_out=buf156, residual=buf144, weight=arg41_1, epsilon=1e-05)
        del arg41_1
        del buf144
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_24], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf160 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf153, arg42_1)
        del arg42_1
        buf161 = buf160
        assert_size_stride(buf161, (s0, 6144), (6144, 1))
        del buf160
        buf162 = reinterpret_tensor(buf153, (s0, 32, 128), (4096, 128, 1), 0); del buf153  # reuse
        # Topologically Sorted Source Nodes: [cat_24], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf161, arg5_1, arg6_1, buf162, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf163 = buf137; del buf137  # reuse
        # Topologically Sorted Source Nodes: [cat_26], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf161, arg5_1, arg6_1, buf163, triton_poi_fused_cat_2_xnumel, stream=stream0)
        buf164 = reinterpret_tensor(buf155, (s0, 32, 128), (4096, 128, 1), 0); del buf155  # reuse
        # Topologically Sorted Source Nodes: [view_45], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf164, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_45, unified_attention_with_output_6], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf162, buf163, reinterpret_tensor(buf161, (s0, 8, 128), (6144, 128, 1), 5120), buf164, 'model.layers.6.self_attn.attn')
        del buf161
        buf167 = reinterpret_tensor(buf162, (s0, 4096), (4096, 1), 0); del buf162  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_25], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf168 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf164, (s0, 4096), (4096, 1), 0), arg43_1)
        del arg43_1
        buf169 = buf168
        assert_size_stride(buf169, (s0, 4096), (4096, 1))
        del buf168
        buf170 = reinterpret_tensor(buf164, (s0, 4096), (4096, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf167, input=buf169, residual_out=buf170, residual=buf156, weight=arg44_1, epsilon=1e-05)
        del arg44_1
        del buf156
        buf174 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_26], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf175 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf167, arg45_1)
        del arg45_1
        buf176 = buf175
        assert_size_stride(buf176, (s0, 28672), (28672, 1))
        del buf175
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf174, buf176)
        del buf176
        buf179 = buf167; del buf167  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_27], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf180 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf174, arg46_1)
        del arg46_1
        buf181 = buf180
        assert_size_stride(buf181, (s0, 4096), (4096, 1))
        del buf180
        buf182 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf179, input=buf181, residual_out=buf182, residual=buf170, weight=arg47_1, epsilon=1e-05)
        del arg47_1
        del buf170
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_28], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf186 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf179, arg48_1)
        del arg48_1
        buf187 = buf186
        assert_size_stride(buf187, (s0, 6144), (6144, 1))
        del buf186
        buf188 = reinterpret_tensor(buf179, (s0, 32, 128), (4096, 128, 1), 0); del buf179  # reuse
        # Topologically Sorted Source Nodes: [cat_28], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf187, arg5_1, arg6_1, buf188, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf189 = buf163; del buf163  # reuse
        # Topologically Sorted Source Nodes: [cat_30], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf187, arg5_1, arg6_1, buf189, triton_poi_fused_cat_2_xnumel, stream=stream0)
        buf190 = reinterpret_tensor(buf181, (s0, 32, 128), (4096, 128, 1), 0); del buf181  # reuse
        # Topologically Sorted Source Nodes: [view_52], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf190, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_52, unified_attention_with_output_7], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf188, buf189, reinterpret_tensor(buf187, (s0, 8, 128), (6144, 128, 1), 5120), buf190, 'model.layers.7.self_attn.attn')
        del buf187
        buf193 = reinterpret_tensor(buf188, (s0, 4096), (4096, 1), 0); del buf188  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_29], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf194 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf190, (s0, 4096), (4096, 1), 0), arg49_1)
        del arg49_1
        buf195 = buf194
        assert_size_stride(buf195, (s0, 4096), (4096, 1))
        del buf194
        buf196 = reinterpret_tensor(buf190, (s0, 4096), (4096, 1), 0); del buf190  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf193, input=buf195, residual_out=buf196, residual=buf182, weight=arg50_1, epsilon=1e-05)
        del arg50_1
        del buf182
        buf200 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_30], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf201 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf193, arg51_1)
        del arg51_1
        buf202 = buf201
        assert_size_stride(buf202, (s0, 28672), (28672, 1))
        del buf201
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf200, buf202)
        del buf202
        buf205 = buf193; del buf193  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_31], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf206 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf200, arg52_1)
        del arg52_1
        buf207 = buf206
        assert_size_stride(buf207, (s0, 4096), (4096, 1))
        del buf206
        buf208 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf205, input=buf207, residual_out=buf208, residual=buf196, weight=arg53_1, epsilon=1e-05)
        del arg53_1
        del buf196
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_32], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf212 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf205, arg54_1)
        del arg54_1
        buf213 = buf212
        assert_size_stride(buf213, (s0, 6144), (6144, 1))
        del buf212
        buf214 = reinterpret_tensor(buf205, (s0, 32, 128), (4096, 128, 1), 0); del buf205  # reuse
        # Topologically Sorted Source Nodes: [cat_32], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf213, arg5_1, arg6_1, buf214, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf215 = buf189; del buf189  # reuse
        # Topologically Sorted Source Nodes: [cat_34], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf213, arg5_1, arg6_1, buf215, triton_poi_fused_cat_2_xnumel, stream=stream0)
        buf216 = reinterpret_tensor(buf207, (s0, 32, 128), (4096, 128, 1), 0); del buf207  # reuse
        # Topologically Sorted Source Nodes: [view_59], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf216, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_59, unified_attention_with_output_8], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf214, buf215, reinterpret_tensor(buf213, (s0, 8, 128), (6144, 128, 1), 5120), buf216, 'model.layers.8.self_attn.attn')
        del buf213
        buf219 = reinterpret_tensor(buf214, (s0, 4096), (4096, 1), 0); del buf214  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_33], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf220 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf216, (s0, 4096), (4096, 1), 0), arg55_1)
        del arg55_1
        buf221 = buf220
        assert_size_stride(buf221, (s0, 4096), (4096, 1))
        del buf220
        buf222 = reinterpret_tensor(buf216, (s0, 4096), (4096, 1), 0); del buf216  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf219, input=buf221, residual_out=buf222, residual=buf208, weight=arg56_1, epsilon=1e-05)
        del arg56_1
        del buf208
        buf226 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_34], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf227 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf219, arg57_1)
        del arg57_1
        buf228 = buf227
        assert_size_stride(buf228, (s0, 28672), (28672, 1))
        del buf227
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf226, buf228)
        del buf228
        buf231 = buf219; del buf219  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_35], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf232 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf226, arg58_1)
        del arg58_1
        buf233 = buf232
        assert_size_stride(buf233, (s0, 4096), (4096, 1))
        del buf232
        buf234 = buf221; del buf221  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf231, input=buf233, residual_out=buf234, residual=buf222, weight=arg59_1, epsilon=1e-05)
        del arg59_1
        del buf222
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_36], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf238 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf231, arg60_1)
        del arg60_1
        buf239 = buf238
        assert_size_stride(buf239, (s0, 6144), (6144, 1))
        del buf238
        buf240 = reinterpret_tensor(buf231, (s0, 32, 128), (4096, 128, 1), 0); del buf231  # reuse
        # Topologically Sorted Source Nodes: [cat_36], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf239, arg5_1, arg6_1, buf240, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf241 = buf215; del buf215  # reuse
        # Topologically Sorted Source Nodes: [cat_38], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf239, arg5_1, arg6_1, buf241, triton_poi_fused_cat_2_xnumel, stream=stream0)
        buf242 = reinterpret_tensor(buf233, (s0, 32, 128), (4096, 128, 1), 0); del buf233  # reuse
        # Topologically Sorted Source Nodes: [view_66], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf242, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_66, unified_attention_with_output_9], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf240, buf241, reinterpret_tensor(buf239, (s0, 8, 128), (6144, 128, 1), 5120), buf242, 'model.layers.9.self_attn.attn')
        del buf239
        buf245 = reinterpret_tensor(buf240, (s0, 4096), (4096, 1), 0); del buf240  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_37], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf246 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf242, (s0, 4096), (4096, 1), 0), arg61_1)
        del arg61_1
        buf247 = buf246
        assert_size_stride(buf247, (s0, 4096), (4096, 1))
        del buf246
        buf248 = reinterpret_tensor(buf242, (s0, 4096), (4096, 1), 0); del buf242  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf245, input=buf247, residual_out=buf248, residual=buf234, weight=arg62_1, epsilon=1e-05)
        del arg62_1
        del buf234
        buf252 = buf226; del buf226  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_38], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf253 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf245, arg63_1)
        del arg63_1
        buf254 = buf253
        assert_size_stride(buf254, (s0, 28672), (28672, 1))
        del buf253
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf252, buf254)
        del buf254
        buf257 = buf245; del buf245  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_39], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf258 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf252, arg64_1)
        del arg64_1
        buf259 = buf258
        assert_size_stride(buf259, (s0, 4096), (4096, 1))
        del buf258
        buf260 = buf247; del buf247  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf257, input=buf259, residual_out=buf260, residual=buf248, weight=arg65_1, epsilon=1e-05)
        del arg65_1
        del buf248
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_40], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf264 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf257, arg66_1)
        del arg66_1
        buf265 = buf264
        assert_size_stride(buf265, (s0, 6144), (6144, 1))
        del buf264
        buf266 = reinterpret_tensor(buf257, (s0, 32, 128), (4096, 128, 1), 0); del buf257  # reuse
        # Topologically Sorted Source Nodes: [cat_40], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf265, arg5_1, arg6_1, buf266, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf267 = buf241; del buf241  # reuse
        # Topologically Sorted Source Nodes: [cat_42], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf265, arg5_1, arg6_1, buf267, triton_poi_fused_cat_2_xnumel, stream=stream0)
        buf268 = reinterpret_tensor(buf259, (s0, 32, 128), (4096, 128, 1), 0); del buf259  # reuse
        # Topologically Sorted Source Nodes: [view_73], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf268, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_73, unified_attention_with_output_10], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf266, buf267, reinterpret_tensor(buf265, (s0, 8, 128), (6144, 128, 1), 5120), buf268, 'model.layers.10.self_attn.attn')
        del buf265
        buf271 = reinterpret_tensor(buf266, (s0, 4096), (4096, 1), 0); del buf266  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_41], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf272 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf268, (s0, 4096), (4096, 1), 0), arg67_1)
        del arg67_1
        buf273 = buf272
        assert_size_stride(buf273, (s0, 4096), (4096, 1))
        del buf272
        buf274 = reinterpret_tensor(buf268, (s0, 4096), (4096, 1), 0); del buf268  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf271, input=buf273, residual_out=buf274, residual=buf260, weight=arg68_1, epsilon=1e-05)
        del arg68_1
        del buf260
        buf278 = buf252; del buf252  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_42], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf279 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf271, arg69_1)
        del arg69_1
        buf280 = buf279
        assert_size_stride(buf280, (s0, 28672), (28672, 1))
        del buf279
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf278, buf280)
        del buf280
        buf283 = buf271; del buf271  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_43], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf284 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf278, arg70_1)
        del arg70_1
        buf285 = buf284
        assert_size_stride(buf285, (s0, 4096), (4096, 1))
        del buf284
        buf286 = buf273; del buf273  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf283, input=buf285, residual_out=buf286, residual=buf274, weight=arg71_1, epsilon=1e-05)
        del arg71_1
        del buf274
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_44], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf290 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf283, arg72_1)
        del arg72_1
        buf291 = buf290
        assert_size_stride(buf291, (s0, 6144), (6144, 1))
        del buf290
        buf292 = reinterpret_tensor(buf283, (s0, 32, 128), (4096, 128, 1), 0); del buf283  # reuse
        # Topologically Sorted Source Nodes: [cat_44], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf291, arg5_1, arg6_1, buf292, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf293 = buf267; del buf267  # reuse
        # Topologically Sorted Source Nodes: [cat_46], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf291, arg5_1, arg6_1, buf293, triton_poi_fused_cat_2_xnumel, stream=stream0)
        buf294 = reinterpret_tensor(buf285, (s0, 32, 128), (4096, 128, 1), 0); del buf285  # reuse
        # Topologically Sorted Source Nodes: [view_80], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf294, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_80, unified_attention_with_output_11], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf292, buf293, reinterpret_tensor(buf291, (s0, 8, 128), (6144, 128, 1), 5120), buf294, 'model.layers.11.self_attn.attn')
        del buf291
        buf297 = reinterpret_tensor(buf292, (s0, 4096), (4096, 1), 0); del buf292  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_45], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf298 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf294, (s0, 4096), (4096, 1), 0), arg73_1)
        del arg73_1
        buf299 = buf298
        assert_size_stride(buf299, (s0, 4096), (4096, 1))
        del buf298
        buf300 = reinterpret_tensor(buf294, (s0, 4096), (4096, 1), 0); del buf294  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf297, input=buf299, residual_out=buf300, residual=buf286, weight=arg74_1, epsilon=1e-05)
        del arg74_1
        del buf286
        buf304 = buf278; del buf278  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_46], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf305 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf297, arg75_1)
        del arg75_1
        buf306 = buf305
        assert_size_stride(buf306, (s0, 28672), (28672, 1))
        del buf305
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf304, buf306)
        del buf306
        buf309 = buf297; del buf297  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_47], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf310 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf304, arg76_1)
        del arg76_1
        buf311 = buf310
        assert_size_stride(buf311, (s0, 4096), (4096, 1))
        del buf310
        buf312 = buf299; del buf299  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf309, input=buf311, residual_out=buf312, residual=buf300, weight=arg77_1, epsilon=1e-05)
        del arg77_1
        del buf300
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_48], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf316 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf309, arg78_1)
        del arg78_1
        buf317 = buf316
        assert_size_stride(buf317, (s0, 6144), (6144, 1))
        del buf316
        buf318 = reinterpret_tensor(buf309, (s0, 32, 128), (4096, 128, 1), 0); del buf309  # reuse
        # Topologically Sorted Source Nodes: [cat_48], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf317, arg5_1, arg6_1, buf318, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf319 = buf293; del buf293  # reuse
        # Topologically Sorted Source Nodes: [cat_50], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf317, arg5_1, arg6_1, buf319, triton_poi_fused_cat_2_xnumel, stream=stream0)
        buf320 = reinterpret_tensor(buf311, (s0, 32, 128), (4096, 128, 1), 0); del buf311  # reuse
        # Topologically Sorted Source Nodes: [view_87], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf320, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_87, unified_attention_with_output_12], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf318, buf319, reinterpret_tensor(buf317, (s0, 8, 128), (6144, 128, 1), 5120), buf320, 'model.layers.12.self_attn.attn')
        del buf317
        buf323 = reinterpret_tensor(buf318, (s0, 4096), (4096, 1), 0); del buf318  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_49], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf324 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf320, (s0, 4096), (4096, 1), 0), arg79_1)
        del arg79_1
        buf325 = buf324
        assert_size_stride(buf325, (s0, 4096), (4096, 1))
        del buf324
        buf326 = reinterpret_tensor(buf320, (s0, 4096), (4096, 1), 0); del buf320  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf323, input=buf325, residual_out=buf326, residual=buf312, weight=arg80_1, epsilon=1e-05)
        del arg80_1
        del buf312
        buf330 = buf304; del buf304  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_50], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf331 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf323, arg81_1)
        del arg81_1
        buf332 = buf331
        assert_size_stride(buf332, (s0, 28672), (28672, 1))
        del buf331
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf330, buf332)
        del buf332
        buf335 = buf323; del buf323  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_51], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf336 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf330, arg82_1)
        del arg82_1
        buf337 = buf336
        assert_size_stride(buf337, (s0, 4096), (4096, 1))
        del buf336
        buf338 = buf325; del buf325  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf335, input=buf337, residual_out=buf338, residual=buf326, weight=arg83_1, epsilon=1e-05)
        del arg83_1
        del buf326
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_52], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf342 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf335, arg84_1)
        del arg84_1
        buf343 = buf342
        assert_size_stride(buf343, (s0, 6144), (6144, 1))
        del buf342
        buf344 = reinterpret_tensor(buf335, (s0, 32, 128), (4096, 128, 1), 0); del buf335  # reuse
        # Topologically Sorted Source Nodes: [cat_52], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf343, arg5_1, arg6_1, buf344, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf345 = buf319; del buf319  # reuse
        # Topologically Sorted Source Nodes: [cat_54], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf343, arg5_1, arg6_1, buf345, triton_poi_fused_cat_2_xnumel, stream=stream0)
        buf346 = reinterpret_tensor(buf337, (s0, 32, 128), (4096, 128, 1), 0); del buf337  # reuse
        # Topologically Sorted Source Nodes: [view_94], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf346, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_94, unified_attention_with_output_13], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf344, buf345, reinterpret_tensor(buf343, (s0, 8, 128), (6144, 128, 1), 5120), buf346, 'model.layers.13.self_attn.attn')
        del buf343
        buf349 = reinterpret_tensor(buf344, (s0, 4096), (4096, 1), 0); del buf344  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_53], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf350 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf346, (s0, 4096), (4096, 1), 0), arg85_1)
        del arg85_1
        buf351 = buf350
        assert_size_stride(buf351, (s0, 4096), (4096, 1))
        del buf350
        buf352 = reinterpret_tensor(buf346, (s0, 4096), (4096, 1), 0); del buf346  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf349, input=buf351, residual_out=buf352, residual=buf338, weight=arg86_1, epsilon=1e-05)
        del arg86_1
        del buf338
        buf356 = buf330; del buf330  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_54], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf357 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf349, arg87_1)
        del arg87_1
        buf358 = buf357
        assert_size_stride(buf358, (s0, 28672), (28672, 1))
        del buf357
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf356, buf358)
        del buf358
        buf361 = buf349; del buf349  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_55], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf362 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf356, arg88_1)
        del arg88_1
        buf363 = buf362
        assert_size_stride(buf363, (s0, 4096), (4096, 1))
        del buf362
        buf364 = buf351; del buf351  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf361, input=buf363, residual_out=buf364, residual=buf352, weight=arg89_1, epsilon=1e-05)
        del arg89_1
        del buf352
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_56], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf368 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf361, arg90_1)
        del arg90_1
        buf369 = buf368
        assert_size_stride(buf369, (s0, 6144), (6144, 1))
        del buf368
        buf370 = reinterpret_tensor(buf361, (s0, 32, 128), (4096, 128, 1), 0); del buf361  # reuse
        # Topologically Sorted Source Nodes: [cat_56], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf369, arg5_1, arg6_1, buf370, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf371 = buf345; del buf345  # reuse
        # Topologically Sorted Source Nodes: [cat_58], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf369, arg5_1, arg6_1, buf371, triton_poi_fused_cat_2_xnumel, stream=stream0)
        buf372 = reinterpret_tensor(buf363, (s0, 32, 128), (4096, 128, 1), 0); del buf363  # reuse
        # Topologically Sorted Source Nodes: [view_101], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf372, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_101, unified_attention_with_output_14], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf370, buf371, reinterpret_tensor(buf369, (s0, 8, 128), (6144, 128, 1), 5120), buf372, 'model.layers.14.self_attn.attn')
        del buf369
        buf375 = reinterpret_tensor(buf370, (s0, 4096), (4096, 1), 0); del buf370  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_57], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf376 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf372, (s0, 4096), (4096, 1), 0), arg91_1)
        del arg91_1
        buf377 = buf376
        assert_size_stride(buf377, (s0, 4096), (4096, 1))
        del buf376
        buf378 = reinterpret_tensor(buf372, (s0, 4096), (4096, 1), 0); del buf372  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf375, input=buf377, residual_out=buf378, residual=buf364, weight=arg92_1, epsilon=1e-05)
        del arg92_1
        del buf364
        buf382 = buf356; del buf356  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_58], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf383 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf375, arg93_1)
        del arg93_1
        buf384 = buf383
        assert_size_stride(buf384, (s0, 28672), (28672, 1))
        del buf383
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf382, buf384)
        del buf384
        buf387 = buf375; del buf375  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_59], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf388 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf382, arg94_1)
        del arg94_1
        buf389 = buf388
        assert_size_stride(buf389, (s0, 4096), (4096, 1))
        del buf388
        buf390 = buf377; del buf377  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf387, input=buf389, residual_out=buf390, residual=buf378, weight=arg95_1, epsilon=1e-05)
        del arg95_1
        del buf378
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_60], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf394 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf387, arg96_1)
        del arg96_1
        buf395 = buf394
        assert_size_stride(buf395, (s0, 6144), (6144, 1))
        del buf394
        buf396 = reinterpret_tensor(buf387, (s0, 32, 128), (4096, 128, 1), 0); del buf387  # reuse
        # Topologically Sorted Source Nodes: [cat_60], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf395, arg5_1, arg6_1, buf396, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf397 = buf371; del buf371  # reuse
        # Topologically Sorted Source Nodes: [cat_62], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf395, arg5_1, arg6_1, buf397, triton_poi_fused_cat_2_xnumel, stream=stream0)
        buf398 = reinterpret_tensor(buf389, (s0, 32, 128), (4096, 128, 1), 0); del buf389  # reuse
        # Topologically Sorted Source Nodes: [view_108], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf398, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_108, unified_attention_with_output_15], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf396, buf397, reinterpret_tensor(buf395, (s0, 8, 128), (6144, 128, 1), 5120), buf398, 'model.layers.15.self_attn.attn')
        del buf395
        buf401 = reinterpret_tensor(buf396, (s0, 4096), (4096, 1), 0); del buf396  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_61], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf402 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf398, (s0, 4096), (4096, 1), 0), arg97_1)
        del arg97_1
        buf403 = buf402
        assert_size_stride(buf403, (s0, 4096), (4096, 1))
        del buf402
        buf404 = reinterpret_tensor(buf398, (s0, 4096), (4096, 1), 0); del buf398  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf401, input=buf403, residual_out=buf404, residual=buf390, weight=arg98_1, epsilon=1e-05)
        del arg98_1
        del buf390
        buf408 = buf382; del buf382  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_62], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf409 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf401, arg99_1)
        del arg99_1
        buf410 = buf409
        assert_size_stride(buf410, (s0, 28672), (28672, 1))
        del buf409
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf408, buf410)
        del buf410
        buf413 = buf401; del buf401  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_63], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf414 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf408, arg100_1)
        del arg100_1
        buf415 = buf414
        assert_size_stride(buf415, (s0, 4096), (4096, 1))
        del buf414
        buf416 = buf403; del buf403  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf413, input=buf415, residual_out=buf416, residual=buf404, weight=arg101_1, epsilon=1e-05)
        del arg101_1
        del buf404
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_64], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf420 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf413, arg102_1)
        del arg102_1
        buf421 = buf420
        assert_size_stride(buf421, (s0, 6144), (6144, 1))
        del buf420
        buf422 = reinterpret_tensor(buf413, (s0, 32, 128), (4096, 128, 1), 0); del buf413  # reuse
        # Topologically Sorted Source Nodes: [cat_64], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf421, arg5_1, arg6_1, buf422, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf423 = buf397; del buf397  # reuse
        # Topologically Sorted Source Nodes: [cat_66], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf421, arg5_1, arg6_1, buf423, triton_poi_fused_cat_2_xnumel, stream=stream0)
        buf424 = reinterpret_tensor(buf415, (s0, 32, 128), (4096, 128, 1), 0); del buf415  # reuse
        # Topologically Sorted Source Nodes: [view_115], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf424, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_115, unified_attention_with_output_16], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf422, buf423, reinterpret_tensor(buf421, (s0, 8, 128), (6144, 128, 1), 5120), buf424, 'model.layers.16.self_attn.attn')
        del buf421
        buf427 = reinterpret_tensor(buf422, (s0, 4096), (4096, 1), 0); del buf422  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_65], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf428 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf424, (s0, 4096), (4096, 1), 0), arg103_1)
        del arg103_1
        buf429 = buf428
        assert_size_stride(buf429, (s0, 4096), (4096, 1))
        del buf428
        buf430 = reinterpret_tensor(buf424, (s0, 4096), (4096, 1), 0); del buf424  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf427, input=buf429, residual_out=buf430, residual=buf416, weight=arg104_1, epsilon=1e-05)
        del arg104_1
        del buf416
        buf434 = buf408; del buf408  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_66], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf435 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf427, arg105_1)
        del arg105_1
        buf436 = buf435
        assert_size_stride(buf436, (s0, 28672), (28672, 1))
        del buf435
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf434, buf436)
        del buf436
        buf439 = buf427; del buf427  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_67], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf440 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf434, arg106_1)
        del arg106_1
        buf441 = buf440
        assert_size_stride(buf441, (s0, 4096), (4096, 1))
        del buf440
        buf442 = buf429; del buf429  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf439, input=buf441, residual_out=buf442, residual=buf430, weight=arg107_1, epsilon=1e-05)
        del arg107_1
        del buf430
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_68], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf446 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf439, arg108_1)
        del arg108_1
        buf447 = buf446
        assert_size_stride(buf447, (s0, 6144), (6144, 1))
        del buf446
        buf448 = reinterpret_tensor(buf439, (s0, 32, 128), (4096, 128, 1), 0); del buf439  # reuse
        # Topologically Sorted Source Nodes: [cat_68], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf447, arg5_1, arg6_1, buf448, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf449 = buf423; del buf423  # reuse
        # Topologically Sorted Source Nodes: [cat_70], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf447, arg5_1, arg6_1, buf449, triton_poi_fused_cat_2_xnumel, stream=stream0)
        buf450 = reinterpret_tensor(buf441, (s0, 32, 128), (4096, 128, 1), 0); del buf441  # reuse
        # Topologically Sorted Source Nodes: [view_122], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf450, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_122, unified_attention_with_output_17], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf448, buf449, reinterpret_tensor(buf447, (s0, 8, 128), (6144, 128, 1), 5120), buf450, 'model.layers.17.self_attn.attn')
        del buf447
        buf453 = reinterpret_tensor(buf448, (s0, 4096), (4096, 1), 0); del buf448  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_69], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf454 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf450, (s0, 4096), (4096, 1), 0), arg109_1)
        del arg109_1
        buf455 = buf454
        assert_size_stride(buf455, (s0, 4096), (4096, 1))
        del buf454
        buf456 = reinterpret_tensor(buf450, (s0, 4096), (4096, 1), 0); del buf450  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf453, input=buf455, residual_out=buf456, residual=buf442, weight=arg110_1, epsilon=1e-05)
        del arg110_1
        del buf442
        buf460 = buf434; del buf434  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_70], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf461 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf453, arg111_1)
        del arg111_1
        buf462 = buf461
        assert_size_stride(buf462, (s0, 28672), (28672, 1))
        del buf461
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf460, buf462)
        del buf462
        buf465 = buf453; del buf453  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_71], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf466 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf460, arg112_1)
        del arg112_1
        buf467 = buf466
        assert_size_stride(buf467, (s0, 4096), (4096, 1))
        del buf466
        buf468 = buf455; del buf455  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf465, input=buf467, residual_out=buf468, residual=buf456, weight=arg113_1, epsilon=1e-05)
        del arg113_1
        del buf456
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_72], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf472 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf465, arg114_1)
        del arg114_1
        buf473 = buf472
        assert_size_stride(buf473, (s0, 6144), (6144, 1))
        del buf472
        buf474 = reinterpret_tensor(buf465, (s0, 32, 128), (4096, 128, 1), 0); del buf465  # reuse
        # Topologically Sorted Source Nodes: [cat_72], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf473, arg5_1, arg6_1, buf474, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf475 = buf449; del buf449  # reuse
        # Topologically Sorted Source Nodes: [cat_74], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf473, arg5_1, arg6_1, buf475, triton_poi_fused_cat_2_xnumel, stream=stream0)
        buf476 = reinterpret_tensor(buf467, (s0, 32, 128), (4096, 128, 1), 0); del buf467  # reuse
        # Topologically Sorted Source Nodes: [view_129], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf476, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_129, unified_attention_with_output_18], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf474, buf475, reinterpret_tensor(buf473, (s0, 8, 128), (6144, 128, 1), 5120), buf476, 'model.layers.18.self_attn.attn')
        del buf473
        buf479 = reinterpret_tensor(buf474, (s0, 4096), (4096, 1), 0); del buf474  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_73], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf480 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf476, (s0, 4096), (4096, 1), 0), arg115_1)
        del arg115_1
        buf481 = buf480
        assert_size_stride(buf481, (s0, 4096), (4096, 1))
        del buf480
        buf482 = reinterpret_tensor(buf476, (s0, 4096), (4096, 1), 0); del buf476  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf479, input=buf481, residual_out=buf482, residual=buf468, weight=arg116_1, epsilon=1e-05)
        del arg116_1
        del buf468
        buf486 = buf460; del buf460  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_74], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf487 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf479, arg117_1)
        del arg117_1
        buf488 = buf487
        assert_size_stride(buf488, (s0, 28672), (28672, 1))
        del buf487
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf486, buf488)
        del buf488
        buf491 = buf479; del buf479  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_75], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf492 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf486, arg118_1)
        del arg118_1
        buf493 = buf492
        assert_size_stride(buf493, (s0, 4096), (4096, 1))
        del buf492
        buf494 = buf481; del buf481  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf491, input=buf493, residual_out=buf494, residual=buf482, weight=arg119_1, epsilon=1e-05)
        del arg119_1
        del buf482
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_76], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf498 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf491, arg120_1)
        del arg120_1
        buf499 = buf498
        assert_size_stride(buf499, (s0, 6144), (6144, 1))
        del buf498
        buf500 = reinterpret_tensor(buf491, (s0, 32, 128), (4096, 128, 1), 0); del buf491  # reuse
        # Topologically Sorted Source Nodes: [cat_76], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf499, arg5_1, arg6_1, buf500, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf501 = buf475; del buf475  # reuse
        # Topologically Sorted Source Nodes: [cat_78], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf499, arg5_1, arg6_1, buf501, triton_poi_fused_cat_2_xnumel, stream=stream0)
        buf502 = reinterpret_tensor(buf493, (s0, 32, 128), (4096, 128, 1), 0); del buf493  # reuse
        # Topologically Sorted Source Nodes: [view_136], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf502, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_136, unified_attention_with_output_19], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf500, buf501, reinterpret_tensor(buf499, (s0, 8, 128), (6144, 128, 1), 5120), buf502, 'model.layers.19.self_attn.attn')
        del buf499
        buf505 = reinterpret_tensor(buf500, (s0, 4096), (4096, 1), 0); del buf500  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_77], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf506 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf502, (s0, 4096), (4096, 1), 0), arg121_1)
        del arg121_1
        buf507 = buf506
        assert_size_stride(buf507, (s0, 4096), (4096, 1))
        del buf506
        buf508 = reinterpret_tensor(buf502, (s0, 4096), (4096, 1), 0); del buf502  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf505, input=buf507, residual_out=buf508, residual=buf494, weight=arg122_1, epsilon=1e-05)
        del arg122_1
        del buf494
        buf512 = buf486; del buf486  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_78], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf513 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf505, arg123_1)
        del arg123_1
        buf514 = buf513
        assert_size_stride(buf514, (s0, 28672), (28672, 1))
        del buf513
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf512, buf514)
        del buf514
        buf517 = buf505; del buf505  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_79], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf518 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf512, arg124_1)
        del arg124_1
        buf519 = buf518
        assert_size_stride(buf519, (s0, 4096), (4096, 1))
        del buf518
        buf520 = buf507; del buf507  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf517, input=buf519, residual_out=buf520, residual=buf508, weight=arg125_1, epsilon=1e-05)
        del arg125_1
        del buf508
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_80], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf524 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf517, arg126_1)
        del arg126_1
        buf525 = buf524
        assert_size_stride(buf525, (s0, 6144), (6144, 1))
        del buf524
        buf526 = reinterpret_tensor(buf517, (s0, 32, 128), (4096, 128, 1), 0); del buf517  # reuse
        # Topologically Sorted Source Nodes: [cat_80], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf525, arg5_1, arg6_1, buf526, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf527 = buf501; del buf501  # reuse
        # Topologically Sorted Source Nodes: [cat_82], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf525, arg5_1, arg6_1, buf527, triton_poi_fused_cat_2_xnumel, stream=stream0)
        buf528 = reinterpret_tensor(buf519, (s0, 32, 128), (4096, 128, 1), 0); del buf519  # reuse
        # Topologically Sorted Source Nodes: [view_143], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf528, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_143, unified_attention_with_output_20], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf526, buf527, reinterpret_tensor(buf525, (s0, 8, 128), (6144, 128, 1), 5120), buf528, 'model.layers.20.self_attn.attn')
        del buf525
        buf531 = reinterpret_tensor(buf526, (s0, 4096), (4096, 1), 0); del buf526  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_81], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf532 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf528, (s0, 4096), (4096, 1), 0), arg127_1)
        del arg127_1
        buf533 = buf532
        assert_size_stride(buf533, (s0, 4096), (4096, 1))
        del buf532
        buf534 = reinterpret_tensor(buf528, (s0, 4096), (4096, 1), 0); del buf528  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf531, input=buf533, residual_out=buf534, residual=buf520, weight=arg128_1, epsilon=1e-05)
        del arg128_1
        del buf520
        buf538 = buf512; del buf512  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_82], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf539 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf531, arg129_1)
        del arg129_1
        buf540 = buf539
        assert_size_stride(buf540, (s0, 28672), (28672, 1))
        del buf539
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf538, buf540)
        del buf540
        buf543 = buf531; del buf531  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_83], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf544 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf538, arg130_1)
        del arg130_1
        buf545 = buf544
        assert_size_stride(buf545, (s0, 4096), (4096, 1))
        del buf544
        buf546 = buf533; del buf533  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf543, input=buf545, residual_out=buf546, residual=buf534, weight=arg131_1, epsilon=1e-05)
        del arg131_1
        del buf534
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_84], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf550 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf543, arg132_1)
        del arg132_1
        buf551 = buf550
        assert_size_stride(buf551, (s0, 6144), (6144, 1))
        del buf550
        buf552 = reinterpret_tensor(buf543, (s0, 32, 128), (4096, 128, 1), 0); del buf543  # reuse
        # Topologically Sorted Source Nodes: [cat_84], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf551, arg5_1, arg6_1, buf552, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf553 = buf527; del buf527  # reuse
        # Topologically Sorted Source Nodes: [cat_86], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf551, arg5_1, arg6_1, buf553, triton_poi_fused_cat_2_xnumel, stream=stream0)
        buf554 = reinterpret_tensor(buf545, (s0, 32, 128), (4096, 128, 1), 0); del buf545  # reuse
        # Topologically Sorted Source Nodes: [view_150], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf554, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_150, unified_attention_with_output_21], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf552, buf553, reinterpret_tensor(buf551, (s0, 8, 128), (6144, 128, 1), 5120), buf554, 'model.layers.21.self_attn.attn')
        del buf551
        buf557 = reinterpret_tensor(buf552, (s0, 4096), (4096, 1), 0); del buf552  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_85], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf558 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf554, (s0, 4096), (4096, 1), 0), arg133_1)
        del arg133_1
        buf559 = buf558
        assert_size_stride(buf559, (s0, 4096), (4096, 1))
        del buf558
        buf560 = reinterpret_tensor(buf554, (s0, 4096), (4096, 1), 0); del buf554  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf557, input=buf559, residual_out=buf560, residual=buf546, weight=arg134_1, epsilon=1e-05)
        del arg134_1
        del buf546
        buf564 = buf538; del buf538  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_86], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf565 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf557, arg135_1)
        del arg135_1
        buf566 = buf565
        assert_size_stride(buf566, (s0, 28672), (28672, 1))
        del buf565
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf564, buf566)
        del buf566
        buf569 = buf557; del buf557  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_87], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf570 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf564, arg136_1)
        del arg136_1
        buf571 = buf570
        assert_size_stride(buf571, (s0, 4096), (4096, 1))
        del buf570
        buf572 = buf559; del buf559  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf569, input=buf571, residual_out=buf572, residual=buf560, weight=arg137_1, epsilon=1e-05)
        del arg137_1
        del buf560
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_88], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf576 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf569, arg138_1)
        del arg138_1
        buf577 = buf576
        assert_size_stride(buf577, (s0, 6144), (6144, 1))
        del buf576
        buf578 = reinterpret_tensor(buf569, (s0, 32, 128), (4096, 128, 1), 0); del buf569  # reuse
        # Topologically Sorted Source Nodes: [cat_88], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf577, arg5_1, arg6_1, buf578, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf579 = buf553; del buf553  # reuse
        # Topologically Sorted Source Nodes: [cat_90], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf577, arg5_1, arg6_1, buf579, triton_poi_fused_cat_2_xnumel, stream=stream0)
        buf580 = reinterpret_tensor(buf571, (s0, 32, 128), (4096, 128, 1), 0); del buf571  # reuse
        # Topologically Sorted Source Nodes: [view_157], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf580, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_157, unified_attention_with_output_22], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf578, buf579, reinterpret_tensor(buf577, (s0, 8, 128), (6144, 128, 1), 5120), buf580, 'model.layers.22.self_attn.attn')
        del buf577
        buf583 = reinterpret_tensor(buf578, (s0, 4096), (4096, 1), 0); del buf578  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_89], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf584 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf580, (s0, 4096), (4096, 1), 0), arg139_1)
        del arg139_1
        buf585 = buf584
        assert_size_stride(buf585, (s0, 4096), (4096, 1))
        del buf584
        buf586 = reinterpret_tensor(buf580, (s0, 4096), (4096, 1), 0); del buf580  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf583, input=buf585, residual_out=buf586, residual=buf572, weight=arg140_1, epsilon=1e-05)
        del arg140_1
        del buf572
        buf590 = buf564; del buf564  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_90], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf591 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf583, arg141_1)
        del arg141_1
        buf592 = buf591
        assert_size_stride(buf592, (s0, 28672), (28672, 1))
        del buf591
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf590, buf592)
        del buf592
        buf595 = buf583; del buf583  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_91], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf596 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf590, arg142_1)
        del arg142_1
        buf597 = buf596
        assert_size_stride(buf597, (s0, 4096), (4096, 1))
        del buf596
        buf598 = buf585; del buf585  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf595, input=buf597, residual_out=buf598, residual=buf586, weight=arg143_1, epsilon=1e-05)
        del arg143_1
        del buf586
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_92], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf602 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf595, arg144_1)
        del arg144_1
        buf603 = buf602
        assert_size_stride(buf603, (s0, 6144), (6144, 1))
        del buf602
        buf604 = reinterpret_tensor(buf595, (s0, 32, 128), (4096, 128, 1), 0); del buf595  # reuse
        # Topologically Sorted Source Nodes: [cat_92], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf603, arg5_1, arg6_1, buf604, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf605 = buf579; del buf579  # reuse
        # Topologically Sorted Source Nodes: [cat_94], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf603, arg5_1, arg6_1, buf605, triton_poi_fused_cat_2_xnumel, stream=stream0)
        buf606 = reinterpret_tensor(buf597, (s0, 32, 128), (4096, 128, 1), 0); del buf597  # reuse
        # Topologically Sorted Source Nodes: [view_164], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf606, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_164, unified_attention_with_output_23], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf604, buf605, reinterpret_tensor(buf603, (s0, 8, 128), (6144, 128, 1), 5120), buf606, 'model.layers.23.self_attn.attn')
        del buf603
        buf609 = reinterpret_tensor(buf604, (s0, 4096), (4096, 1), 0); del buf604  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_93], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf610 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf606, (s0, 4096), (4096, 1), 0), arg145_1)
        del arg145_1
        buf611 = buf610
        assert_size_stride(buf611, (s0, 4096), (4096, 1))
        del buf610
        buf612 = reinterpret_tensor(buf606, (s0, 4096), (4096, 1), 0); del buf606  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf609, input=buf611, residual_out=buf612, residual=buf598, weight=arg146_1, epsilon=1e-05)
        del arg146_1
        del buf598
        buf616 = buf590; del buf590  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_94], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf617 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf609, arg147_1)
        del arg147_1
        buf618 = buf617
        assert_size_stride(buf618, (s0, 28672), (28672, 1))
        del buf617
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf616, buf618)
        del buf618
        buf621 = buf609; del buf609  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_95], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf622 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf616, arg148_1)
        del arg148_1
        buf623 = buf622
        assert_size_stride(buf623, (s0, 4096), (4096, 1))
        del buf622
        buf624 = buf611; del buf611  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf621, input=buf623, residual_out=buf624, residual=buf612, weight=arg149_1, epsilon=1e-05)
        del arg149_1
        del buf612
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_96], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf628 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf621, arg150_1)
        del arg150_1
        buf629 = buf628
        assert_size_stride(buf629, (s0, 6144), (6144, 1))
        del buf628
        buf630 = reinterpret_tensor(buf621, (s0, 32, 128), (4096, 128, 1), 0); del buf621  # reuse
        # Topologically Sorted Source Nodes: [cat_96], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf629, arg5_1, arg6_1, buf630, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf631 = buf605; del buf605  # reuse
        # Topologically Sorted Source Nodes: [cat_98], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf629, arg5_1, arg6_1, buf631, triton_poi_fused_cat_2_xnumel, stream=stream0)
        buf632 = reinterpret_tensor(buf623, (s0, 32, 128), (4096, 128, 1), 0); del buf623  # reuse
        # Topologically Sorted Source Nodes: [view_171], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf632, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_171, unified_attention_with_output_24], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf630, buf631, reinterpret_tensor(buf629, (s0, 8, 128), (6144, 128, 1), 5120), buf632, 'model.layers.24.self_attn.attn')
        del buf629
        buf635 = reinterpret_tensor(buf630, (s0, 4096), (4096, 1), 0); del buf630  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_97], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf636 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf632, (s0, 4096), (4096, 1), 0), arg151_1)
        del arg151_1
        buf637 = buf636
        assert_size_stride(buf637, (s0, 4096), (4096, 1))
        del buf636
        buf638 = reinterpret_tensor(buf632, (s0, 4096), (4096, 1), 0); del buf632  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf635, input=buf637, residual_out=buf638, residual=buf624, weight=arg152_1, epsilon=1e-05)
        del arg152_1
        del buf624
        buf642 = buf616; del buf616  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_98], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf643 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf635, arg153_1)
        del arg153_1
        buf644 = buf643
        assert_size_stride(buf644, (s0, 28672), (28672, 1))
        del buf643
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf642, buf644)
        del buf644
        buf647 = buf635; del buf635  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_99], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf648 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf642, arg154_1)
        del arg154_1
        buf649 = buf648
        assert_size_stride(buf649, (s0, 4096), (4096, 1))
        del buf648
        buf650 = buf637; del buf637  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf647, input=buf649, residual_out=buf650, residual=buf638, weight=arg155_1, epsilon=1e-05)
        del arg155_1
        del buf638
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_100], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf654 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf647, arg156_1)
        del arg156_1
        buf655 = buf654
        assert_size_stride(buf655, (s0, 6144), (6144, 1))
        del buf654
        buf656 = reinterpret_tensor(buf647, (s0, 32, 128), (4096, 128, 1), 0); del buf647  # reuse
        # Topologically Sorted Source Nodes: [cat_100], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf655, arg5_1, arg6_1, buf656, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf657 = buf631; del buf631  # reuse
        # Topologically Sorted Source Nodes: [cat_102], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf655, arg5_1, arg6_1, buf657, triton_poi_fused_cat_2_xnumel, stream=stream0)
        buf658 = reinterpret_tensor(buf649, (s0, 32, 128), (4096, 128, 1), 0); del buf649  # reuse
        # Topologically Sorted Source Nodes: [view_178], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf658, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_178, unified_attention_with_output_25], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf656, buf657, reinterpret_tensor(buf655, (s0, 8, 128), (6144, 128, 1), 5120), buf658, 'model.layers.25.self_attn.attn')
        del buf655
        buf661 = reinterpret_tensor(buf656, (s0, 4096), (4096, 1), 0); del buf656  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_101], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf662 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf658, (s0, 4096), (4096, 1), 0), arg157_1)
        del arg157_1
        buf663 = buf662
        assert_size_stride(buf663, (s0, 4096), (4096, 1))
        del buf662
        buf664 = reinterpret_tensor(buf658, (s0, 4096), (4096, 1), 0); del buf658  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf661, input=buf663, residual_out=buf664, residual=buf650, weight=arg158_1, epsilon=1e-05)
        del arg158_1
        del buf650
        buf668 = buf642; del buf642  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_102], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf669 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf661, arg159_1)
        del arg159_1
        buf670 = buf669
        assert_size_stride(buf670, (s0, 28672), (28672, 1))
        del buf669
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf668, buf670)
        del buf670
        buf673 = buf661; del buf661  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_103], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf674 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf668, arg160_1)
        del arg160_1
        buf675 = buf674
        assert_size_stride(buf675, (s0, 4096), (4096, 1))
        del buf674
        buf676 = buf663; del buf663  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf673, input=buf675, residual_out=buf676, residual=buf664, weight=arg161_1, epsilon=1e-05)
        del arg161_1
        del buf664
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_104], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf680 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf673, arg162_1)
        del arg162_1
        buf681 = buf680
        assert_size_stride(buf681, (s0, 6144), (6144, 1))
        del buf680
        buf682 = reinterpret_tensor(buf673, (s0, 32, 128), (4096, 128, 1), 0); del buf673  # reuse
        # Topologically Sorted Source Nodes: [cat_104], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf681, arg5_1, arg6_1, buf682, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf683 = buf657; del buf657  # reuse
        # Topologically Sorted Source Nodes: [cat_106], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf681, arg5_1, arg6_1, buf683, triton_poi_fused_cat_2_xnumel, stream=stream0)
        buf684 = reinterpret_tensor(buf675, (s0, 32, 128), (4096, 128, 1), 0); del buf675  # reuse
        # Topologically Sorted Source Nodes: [view_185], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf684, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_185, unified_attention_with_output_26], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf682, buf683, reinterpret_tensor(buf681, (s0, 8, 128), (6144, 128, 1), 5120), buf684, 'model.layers.26.self_attn.attn')
        del buf681
        buf687 = reinterpret_tensor(buf682, (s0, 4096), (4096, 1), 0); del buf682  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_105], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf688 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf684, (s0, 4096), (4096, 1), 0), arg163_1)
        del arg163_1
        buf689 = buf688
        assert_size_stride(buf689, (s0, 4096), (4096, 1))
        del buf688
        buf690 = reinterpret_tensor(buf684, (s0, 4096), (4096, 1), 0); del buf684  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf687, input=buf689, residual_out=buf690, residual=buf676, weight=arg164_1, epsilon=1e-05)
        del arg164_1
        del buf676
        buf694 = buf668; del buf668  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_106], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf695 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf687, arg165_1)
        del arg165_1
        buf696 = buf695
        assert_size_stride(buf696, (s0, 28672), (28672, 1))
        del buf695
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf694, buf696)
        del buf696
        buf699 = buf687; del buf687  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_107], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf700 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf694, arg166_1)
        del arg166_1
        buf701 = buf700
        assert_size_stride(buf701, (s0, 4096), (4096, 1))
        del buf700
        buf702 = buf689; del buf689  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf699, input=buf701, residual_out=buf702, residual=buf690, weight=arg167_1, epsilon=1e-05)
        del arg167_1
        del buf690
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_108], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf706 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf699, arg168_1)
        del arg168_1
        buf707 = buf706
        assert_size_stride(buf707, (s0, 6144), (6144, 1))
        del buf706
        buf708 = reinterpret_tensor(buf699, (s0, 32, 128), (4096, 128, 1), 0); del buf699  # reuse
        # Topologically Sorted Source Nodes: [cat_108], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf707, arg5_1, arg6_1, buf708, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf709 = buf683; del buf683  # reuse
        # Topologically Sorted Source Nodes: [cat_110], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf707, arg5_1, arg6_1, buf709, triton_poi_fused_cat_2_xnumel, stream=stream0)
        buf710 = reinterpret_tensor(buf701, (s0, 32, 128), (4096, 128, 1), 0); del buf701  # reuse
        # Topologically Sorted Source Nodes: [view_192], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf710, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_192, unified_attention_with_output_27], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf708, buf709, reinterpret_tensor(buf707, (s0, 8, 128), (6144, 128, 1), 5120), buf710, 'model.layers.27.self_attn.attn')
        del buf707
        buf713 = reinterpret_tensor(buf708, (s0, 4096), (4096, 1), 0); del buf708  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_109], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf714 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf710, (s0, 4096), (4096, 1), 0), arg169_1)
        del arg169_1
        buf715 = buf714
        assert_size_stride(buf715, (s0, 4096), (4096, 1))
        del buf714
        buf716 = reinterpret_tensor(buf710, (s0, 4096), (4096, 1), 0); del buf710  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf713, input=buf715, residual_out=buf716, residual=buf702, weight=arg170_1, epsilon=1e-05)
        del arg170_1
        del buf702
        buf720 = buf694; del buf694  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_110], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf721 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf713, arg171_1)
        del arg171_1
        buf722 = buf721
        assert_size_stride(buf722, (s0, 28672), (28672, 1))
        del buf721
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf720, buf722)
        del buf722
        buf725 = buf713; del buf713  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_111], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf726 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf720, arg172_1)
        del arg172_1
        buf727 = buf726
        assert_size_stride(buf727, (s0, 4096), (4096, 1))
        del buf726
        buf728 = buf715; del buf715  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf725, input=buf727, residual_out=buf728, residual=buf716, weight=arg173_1, epsilon=1e-05)
        del arg173_1
        del buf716
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_112], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf732 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf725, arg174_1)
        del arg174_1
        buf733 = buf732
        assert_size_stride(buf733, (s0, 6144), (6144, 1))
        del buf732
        buf734 = reinterpret_tensor(buf725, (s0, 32, 128), (4096, 128, 1), 0); del buf725  # reuse
        # Topologically Sorted Source Nodes: [cat_112], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf733, arg5_1, arg6_1, buf734, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf735 = buf709; del buf709  # reuse
        # Topologically Sorted Source Nodes: [cat_114], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf733, arg5_1, arg6_1, buf735, triton_poi_fused_cat_2_xnumel, stream=stream0)
        buf736 = reinterpret_tensor(buf727, (s0, 32, 128), (4096, 128, 1), 0); del buf727  # reuse
        # Topologically Sorted Source Nodes: [view_199], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf736, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_199, unified_attention_with_output_28], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf734, buf735, reinterpret_tensor(buf733, (s0, 8, 128), (6144, 128, 1), 5120), buf736, 'model.layers.28.self_attn.attn')
        del buf733
        buf739 = reinterpret_tensor(buf734, (s0, 4096), (4096, 1), 0); del buf734  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_113], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf740 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf736, (s0, 4096), (4096, 1), 0), arg175_1)
        del arg175_1
        buf741 = buf740
        assert_size_stride(buf741, (s0, 4096), (4096, 1))
        del buf740
        buf742 = reinterpret_tensor(buf736, (s0, 4096), (4096, 1), 0); del buf736  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf739, input=buf741, residual_out=buf742, residual=buf728, weight=arg176_1, epsilon=1e-05)
        del arg176_1
        del buf728
        buf746 = buf720; del buf720  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_114], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf747 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf739, arg177_1)
        del arg177_1
        buf748 = buf747
        assert_size_stride(buf748, (s0, 28672), (28672, 1))
        del buf747
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf746, buf748)
        del buf748
        buf751 = buf739; del buf739  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_115], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf752 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf746, arg178_1)
        del arg178_1
        buf753 = buf752
        assert_size_stride(buf753, (s0, 4096), (4096, 1))
        del buf752
        buf754 = buf741; del buf741  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf751, input=buf753, residual_out=buf754, residual=buf742, weight=arg179_1, epsilon=1e-05)
        del arg179_1
        del buf742
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_116], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf758 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf751, arg180_1)
        del arg180_1
        buf759 = buf758
        assert_size_stride(buf759, (s0, 6144), (6144, 1))
        del buf758
        buf760 = reinterpret_tensor(buf751, (s0, 32, 128), (4096, 128, 1), 0); del buf751  # reuse
        # Topologically Sorted Source Nodes: [cat_116], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf759, arg5_1, arg6_1, buf760, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf761 = buf735; del buf735  # reuse
        # Topologically Sorted Source Nodes: [cat_118], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf759, arg5_1, arg6_1, buf761, triton_poi_fused_cat_2_xnumel, stream=stream0)
        buf762 = reinterpret_tensor(buf753, (s0, 32, 128), (4096, 128, 1), 0); del buf753  # reuse
        # Topologically Sorted Source Nodes: [view_206], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf762, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_206, unified_attention_with_output_29], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf760, buf761, reinterpret_tensor(buf759, (s0, 8, 128), (6144, 128, 1), 5120), buf762, 'model.layers.29.self_attn.attn')
        del buf759
        buf765 = reinterpret_tensor(buf760, (s0, 4096), (4096, 1), 0); del buf760  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_117], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf766 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf762, (s0, 4096), (4096, 1), 0), arg181_1)
        del arg181_1
        buf767 = buf766
        assert_size_stride(buf767, (s0, 4096), (4096, 1))
        del buf766
        buf768 = reinterpret_tensor(buf762, (s0, 4096), (4096, 1), 0); del buf762  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf765, input=buf767, residual_out=buf768, residual=buf754, weight=arg182_1, epsilon=1e-05)
        del arg182_1
        del buf754
        buf772 = buf746; del buf746  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_118], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf773 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf765, arg183_1)
        del arg183_1
        buf774 = buf773
        assert_size_stride(buf774, (s0, 28672), (28672, 1))
        del buf773
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf772, buf774)
        del buf774
        buf777 = buf765; del buf765  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_119], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf778 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf772, arg184_1)
        del arg184_1
        buf779 = buf778
        assert_size_stride(buf779, (s0, 4096), (4096, 1))
        del buf778
        buf780 = buf767; del buf767  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf777, input=buf779, residual_out=buf780, residual=buf768, weight=arg185_1, epsilon=1e-05)
        del arg185_1
        del buf768
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_120], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf784 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf777, arg186_1)
        del arg186_1
        buf785 = buf784
        assert_size_stride(buf785, (s0, 6144), (6144, 1))
        del buf784
        buf786 = reinterpret_tensor(buf777, (s0, 32, 128), (4096, 128, 1), 0); del buf777  # reuse
        # Topologically Sorted Source Nodes: [cat_120], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf785, arg5_1, arg6_1, buf786, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf787 = buf761; del buf761  # reuse
        # Topologically Sorted Source Nodes: [cat_122], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf785, arg5_1, arg6_1, buf787, triton_poi_fused_cat_2_xnumel, stream=stream0)
        buf788 = reinterpret_tensor(buf779, (s0, 32, 128), (4096, 128, 1), 0); del buf779  # reuse
        # Topologically Sorted Source Nodes: [view_213], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf788, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_213, unified_attention_with_output_30], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf786, buf787, reinterpret_tensor(buf785, (s0, 8, 128), (6144, 128, 1), 5120), buf788, 'model.layers.30.self_attn.attn')
        del buf785
        buf791 = reinterpret_tensor(buf786, (s0, 4096), (4096, 1), 0); del buf786  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_121], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf792 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf788, (s0, 4096), (4096, 1), 0), arg187_1)
        del arg187_1
        buf793 = buf792
        assert_size_stride(buf793, (s0, 4096), (4096, 1))
        del buf792
        buf794 = reinterpret_tensor(buf788, (s0, 4096), (4096, 1), 0); del buf788  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf791, input=buf793, residual_out=buf794, residual=buf780, weight=arg188_1, epsilon=1e-05)
        del arg188_1
        del buf780
        buf798 = buf772; del buf772  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_122], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf799 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf791, arg189_1)
        del arg189_1
        buf800 = buf799
        assert_size_stride(buf800, (s0, 28672), (28672, 1))
        del buf799
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf798, buf800)
        del buf800
        buf803 = buf791; del buf791  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_123], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf804 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf798, arg190_1)
        del arg190_1
        buf805 = buf804
        assert_size_stride(buf805, (s0, 4096), (4096, 1))
        del buf804
        buf806 = buf793; del buf793  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf803, input=buf805, residual_out=buf806, residual=buf794, weight=arg191_1, epsilon=1e-05)
        del arg191_1
        del buf794
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_124], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf810 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf803, arg192_1)
        del arg192_1
        buf811 = buf810
        assert_size_stride(buf811, (s0, 6144), (6144, 1))
        del buf810
        buf812 = reinterpret_tensor(buf803, (s0, 32, 128), (4096, 128, 1), 0); del buf803  # reuse
        # Topologically Sorted Source Nodes: [cat_124], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf811, arg5_1, arg6_1, buf812, triton_poi_fused_cat_1_xnumel, stream=stream0)
        buf813 = buf787; del buf787  # reuse
        # Topologically Sorted Source Nodes: [cat_126], Original ATen: [aten.cat]
        triton_poi_fused_cat_2_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf811, arg5_1, arg6_1, buf813, triton_poi_fused_cat_2_xnumel, stream=stream0)
        del arg5_1
        del arg6_1
        buf814 = reinterpret_tensor(buf805, (s0, 32, 128), (4096, 128, 1), 0); del buf805  # reuse
        # Topologically Sorted Source Nodes: [view_220], Original ATen: [aten.view]
        triton_poi_fused_view_3_xnumel = 4096*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf814, triton_poi_fused_view_3_xnumel, stream=stream0)
        # Topologically Sorted Source Nodes: [view_220, unified_attention_with_output_31], Original ATen: [aten.view]
        torch.ops.vllm.unified_attention_with_output.default(buf812, buf813, reinterpret_tensor(buf811, (s0, 8, 128), (6144, 128, 1), 5120), buf814, 'model.layers.31.self_attn.attn')
        del buf811
        del buf813
        buf817 = reinterpret_tensor(buf812, (s0, 4096), (4096, 1), 0); del buf812  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_125], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf818 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(reinterpret_tensor(buf814, (s0, 4096), (4096, 1), 0), arg193_1)
        del arg193_1
        buf819 = buf818
        assert_size_stride(buf819, (s0, 4096), (4096, 1))
        del buf818
        buf820 = reinterpret_tensor(buf814, (s0, 4096), (4096, 1), 0); del buf814  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf817, input=buf819, residual_out=buf820, residual=buf806, weight=arg194_1, epsilon=1e-05)
        del arg194_1
        del buf806
        buf824 = buf798; del buf798  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_126], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf825 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf817, arg195_1)
        del arg195_1
        buf826 = buf825
        assert_size_stride(buf826, (s0, 28672), (28672, 1))
        del buf825
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.silu_and_mul.default(buf824, buf826)
        del buf826
        buf829 = buf817; del buf817  # reuse
        # Topologically Sorted Source Nodes: [rocm_unquantized_gemm_impl_127], Original ATen: [vllm.rocm_unquantized_gemm_impl]
        buf830 = torch.ops.vllm.rocm_unquantized_gemm_impl.default(buf824, arg196_1)
        del arg196_1
        del buf824
        buf831 = buf830
        assert_size_stride(buf831, (s0, 4096), (4096, 1))
        del buf830
        buf832 = buf819; del buf819  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.fused_add_rms_norm.default(result=buf829, input=buf831, residual_out=buf832, residual=buf820, weight=arg197_1, epsilon=1e-05)
        del arg197_1
        del buf820
        del buf831
        del buf832
    return (buf829, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg1_1 = 8192
    arg2_1 = rand_strided((128256, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg6_1 = rand_strided((131072, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg11_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg12_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg13_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg14_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg15_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg16_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg17_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg18_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg19_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg20_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg21_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg22_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg23_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg24_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg25_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg26_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg27_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg28_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg29_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg30_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg31_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg32_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg33_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg34_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg35_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg36_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg37_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg38_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg39_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg40_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg41_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg42_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg43_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg44_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg45_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg46_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg47_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg48_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg49_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg50_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg51_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg52_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg53_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg54_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg55_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg56_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg57_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg58_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg59_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg60_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg61_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg62_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg63_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg64_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg65_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg66_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg67_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg68_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg69_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg70_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg71_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg72_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg73_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg74_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg75_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg76_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg77_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg78_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg79_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg80_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg81_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg82_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg83_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg84_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg85_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg86_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg87_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg88_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg89_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg90_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg91_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg92_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg93_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg94_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg95_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg96_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg97_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg98_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg99_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg100_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg101_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg102_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg103_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg104_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg105_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg106_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg107_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg108_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg109_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg110_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg111_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg112_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg113_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg114_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg115_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg116_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg117_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg118_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg119_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg120_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg121_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg122_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg123_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg124_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg125_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg126_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg127_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg128_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg129_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg130_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg131_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg132_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg133_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg134_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg135_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg136_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg137_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg138_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg139_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg140_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg141_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg142_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg143_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg144_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg145_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg146_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg147_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg148_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg149_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg150_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg151_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg152_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg153_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg154_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg155_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg156_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg157_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg158_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg159_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg160_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg161_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg162_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg163_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg164_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg165_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg166_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg167_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg168_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg169_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg170_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg171_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg172_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg173_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg174_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg175_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg176_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg177_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg178_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg179_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg180_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg181_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg182_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg183_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg184_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg185_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg186_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg187_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg188_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg189_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg190_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg191_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg192_1 = rand_strided((6144, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg193_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg194_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg195_1 = rand_strided((28672, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg196_1 = rand_strided((4096, 14336), (14336, 1), device='cuda:0', dtype=torch.bfloat16)
    arg197_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
