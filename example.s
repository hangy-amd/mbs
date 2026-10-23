
# __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa--gfx950
	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z20read_pattern_kernel2PKhmPh ; -- Begin function _Z20read_pattern_kernel2PKhmPh
	.globl	_Z20read_pattern_kernel2PKhmPh
	.p2align	8
	.type	_Z20read_pattern_kernel2PKhmPh,@function
_Z20read_pattern_kernel2PKhmPh:         ; @_Z20read_pattern_kernel2PKhmPh
; %bb.0:
	s_load_dwordx4 s[4:7], s[0:1], 0x0
	s_load_dwordx2 s[8:9], s[0:1], 0x10
	v_lshlrev_b32_e32 v0, 4, v0
	v_mov_b32_e32 v1, 0
	v_mov_b32_e32 v2, s2
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s3, s7, s2
	v_mad_u64_u32 v[16:17], s[0:1], s6, v2, v[0:1]
	v_add_u32_e32 v17, s3, v17
	v_lshl_add_u64 v[18:19], s[4:5], 0, v[16:17]
	global_load_dwordx4 v[0:3], v[18:19], off
	global_load_dwordx4 v[4:7], v[18:19], off offset:1024
	global_load_dwordx4 v[8:11], v[18:19], off offset:2048
	global_load_dwordx4 v[12:15], v[18:19], off offset:3072
	v_lshl_add_u64 v[16:17], s[8:9], 0, v[16:17]
	s_waitcnt vmcnt(3)
	global_store_dwordx4 v[16:17], v[0:3], off
	s_waitcnt vmcnt(3)
	global_store_dwordx4 v[16:17], v[4:7], off offset:1024
	s_waitcnt vmcnt(3)
	global_store_dwordx4 v[16:17], v[8:11], off offset:2048
	s_waitcnt vmcnt(3)
	global_store_dwordx4 v[16:17], v[12:15], off offset:3072
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z20read_pattern_kernel2PKhmPh
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 20
		.amdhsa_next_free_sgpr 10
		.amdhsa_accum_offset 20
		.amdhsa_reserve_vcc 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	_Z20read_pattern_kernel2PKhmPh, .Lfunc_end0-_Z20read_pattern_kernel2PKhmPh
                                        ; -- End function
	.set _Z20read_pattern_kernel2PKhmPh.num_vgpr, 20
	.set _Z20read_pattern_kernel2PKhmPh.num_agpr, 0
	.set _Z20read_pattern_kernel2PKhmPh.numbered_sgpr, 10
	.set _Z20read_pattern_kernel2PKhmPh.num_named_barrier, 0
	.set _Z20read_pattern_kernel2PKhmPh.private_seg_size, 0
	.set _Z20read_pattern_kernel2PKhmPh.uses_vcc, 0
	.set _Z20read_pattern_kernel2PKhmPh.uses_flat_scratch, 0
	.set _Z20read_pattern_kernel2PKhmPh.has_dyn_sized_stack, 0
	.set _Z20read_pattern_kernel2PKhmPh.has_recursion, 0
	.set _Z20read_pattern_kernel2PKhmPh.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 148
; TotalNumSgprs: 16
; NumVgprs: 20
; NumAgprs: 0
; TotalNumVgprs: 20
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 16
; NumVGPRsForWavesPerEU: 20
; AccumOffset: 20
; Occupancy: 8
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 4
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_6f79350ec0a424b5,@object ; @__hip_cuid_6f79350ec0a424b5
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_6f79350ec0a424b5
__hip_cuid_6f79350ec0a424b5:
	.byte	0                               ; 0x0
	.size	__hip_cuid_6f79350ec0a424b5, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.3 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_6f79350ec0a424b5
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           8
        .value_kind:     by_value
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z20read_pattern_kernel2PKhmPh
    .private_segment_fixed_size: 0
    .sgpr_count:     16
    .sgpr_spill_count: 0
    .symbol:         _Z20read_pattern_kernel2PKhmPh.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     20
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

# __CLANG_OFFLOAD_BUNDLE____END__ hip-amdgcn-amd-amdhsa--gfx950

# __CLANG_OFFLOAD_BUNDLE____START__ host-x86_64-unknown-linux-gnu-
	.file	"example.cpp"
                                        # Start of file scope inline assembly
	.globl	_ZSt21ios_base_library_initv

                                        # End of file scope inline assembly
	.section	.text._Z10tensor_addIfEvPKT_S2_PS0_l,"axG",@progbits,_Z10tensor_addIfEvPKT_S2_PS0_l,comdat
	.weak	_Z10tensor_addIfEvPKT_S2_PS0_l  # -- Begin function _Z10tensor_addIfEvPKT_S2_PS0_l
	.p2align	4
	.type	_Z10tensor_addIfEvPKT_S2_PS0_l,@function
_Z10tensor_addIfEvPKT_S2_PS0_l:         # @_Z10tensor_addIfEvPKT_S2_PS0_l
	.cfi_startproc
# %bb.0:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset %rbx, -16
	testq	%rcx, %rcx
	jle	.LBB0_13
# %bb.1:
	cmpq	$8, %rcx
	jb	.LBB0_2
# %bb.3:
	movq	%rdx, %rax
	subq	%rdi, %rax
	cmpq	$32, %rax
	setb	%al
	movq	%rdx, %r8
	subq	%rsi, %r8
	cmpq	$32, %r8
	setb	%r8b
	orb	%al, %r8b
	je	.LBB0_5
.LBB0_2:
	xorl	%eax, %eax
.LBB0_8:
	movq	%rcx, %r9
	movq	%rax, %r8
	andq	$3, %r9
	je	.LBB0_11
# %bb.9:
	movq	%rax, %r8
	.p2align	4
.LBB0_10:                               # =>This Inner Loop Header: Depth=1
	movss	(%rdi,%r8,4), %xmm0             # xmm0 = mem[0],zero,zero,zero
	addss	(%rsi,%r8,4), %xmm0
	movss	%xmm0, (%rdx,%r8,4)
	incq	%r8
	decq	%r9
	jne	.LBB0_10
.LBB0_11:
	subq	%rcx, %rax
	cmpq	$-4, %rax
	ja	.LBB0_13
	.p2align	4
.LBB0_12:                               # =>This Inner Loop Header: Depth=1
	movss	(%rdi,%r8,4), %xmm0             # xmm0 = mem[0],zero,zero,zero
	addss	(%rsi,%r8,4), %xmm0
	movss	%xmm0, (%rdx,%r8,4)
	movss	4(%rdi,%r8,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	addss	4(%rsi,%r8,4), %xmm0
	movss	%xmm0, 4(%rdx,%r8,4)
	movss	8(%rdi,%r8,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	addss	8(%rsi,%r8,4), %xmm0
	movss	%xmm0, 8(%rdx,%r8,4)
	movss	12(%rdi,%r8,4), %xmm0           # xmm0 = mem[0],zero,zero,zero
	addss	12(%rsi,%r8,4), %xmm0
	movss	%xmm0, 12(%rdx,%r8,4)
	addq	$4, %r8
	cmpq	%r8, %rcx
	jne	.LBB0_12
	jmp	.LBB0_13
.LBB0_5:
	movabsq	$9223372036854775800, %rax      # imm = 0x7FFFFFFFFFFFFFF8
	andq	%rcx, %rax
	xorl	%r8d, %r8d
	.p2align	4
.LBB0_6:                                # =>This Inner Loop Header: Depth=1
	movups	(%rdi,%r8,4), %xmm0
	movups	16(%rdi,%r8,4), %xmm1
	movups	(%rsi,%r8,4), %xmm2
	addps	%xmm0, %xmm2
	movups	16(%rsi,%r8,4), %xmm0
	addps	%xmm1, %xmm0
	movups	%xmm2, (%rdx,%r8,4)
	movups	%xmm0, 16(%rdx,%r8,4)
	addq	$8, %r8
	cmpq	%r8, %rax
	jne	.LBB0_6
# %bb.7:
	cmpq	%rax, %rcx
	jne	.LBB0_8
.LBB0_13:
	movl	$_ZSt4cout, %edi
	movl	$.L.str, %esi
	movl	$20, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	_ZSt4cout(%rip), %rax
	movq	-24(%rax), %rax
	movq	_ZSt4cout+240(%rax), %rbx
	testq	%rbx, %rbx
	je	.LBB0_18
# %bb.14:
	cmpb	$0, 56(%rbx)
	je	.LBB0_16
# %bb.15:
	movzbl	67(%rbx), %eax
	jmp	.LBB0_17
.LBB0_16:
	movq	%rbx, %rdi
	callq	_ZNKSt5ctypeIcE13_M_widen_initEv
	movq	(%rbx), %rax
	movq	%rbx, %rdi
	movl	$10, %esi
	callq	*48(%rax)
.LBB0_17:
	movsbl	%al, %esi
	movl	$_ZSt4cout, %edi
	callq	_ZNSo3putEc
	movq	%rax, %rdi
	popq	%rbx
	.cfi_def_cfa_offset 8
	jmp	_ZNSo5flushEv                   # TAILCALL
.LBB0_18:
	.cfi_def_cfa_offset 16
	callq	_ZSt16__throw_bad_castv
.Lfunc_end0:
	.size	_Z10tensor_addIfEvPKT_S2_PS0_l, .Lfunc_end0-_Z10tensor_addIfEvPKT_S2_PS0_l
	.cfi_endproc
                                        # -- End function
	.section	.text._Z10tensor_addIdEvPKT_S2_PS0_l,"axG",@progbits,_Z10tensor_addIdEvPKT_S2_PS0_l,comdat
	.weak	_Z10tensor_addIdEvPKT_S2_PS0_l  # -- Begin function _Z10tensor_addIdEvPKT_S2_PS0_l
	.p2align	4
	.type	_Z10tensor_addIdEvPKT_S2_PS0_l,@function
_Z10tensor_addIdEvPKT_S2_PS0_l:         # @_Z10tensor_addIdEvPKT_S2_PS0_l
	.cfi_startproc
# %bb.0:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset %rbx, -16
	testq	%rcx, %rcx
	jle	.LBB1_13
# %bb.1:
	cmpq	$8, %rcx
	jb	.LBB1_2
# %bb.3:
	movq	%rdx, %rax
	subq	%rdi, %rax
	cmpq	$32, %rax
	setb	%al
	movq	%rdx, %r8
	subq	%rsi, %r8
	cmpq	$32, %r8
	setb	%r8b
	orb	%al, %r8b
	je	.LBB1_5
.LBB1_2:
	xorl	%eax, %eax
.LBB1_8:
	movq	%rcx, %r9
	movq	%rax, %r8
	andq	$3, %r9
	je	.LBB1_11
# %bb.9:
	movq	%rax, %r8
	.p2align	4
.LBB1_10:                               # =>This Inner Loop Header: Depth=1
	movsd	(%rdi,%r8,8), %xmm0             # xmm0 = mem[0],zero
	addsd	(%rsi,%r8,8), %xmm0
	movsd	%xmm0, (%rdx,%r8,8)
	incq	%r8
	decq	%r9
	jne	.LBB1_10
.LBB1_11:
	subq	%rcx, %rax
	cmpq	$-4, %rax
	ja	.LBB1_13
	.p2align	4
.LBB1_12:                               # =>This Inner Loop Header: Depth=1
	movsd	(%rdi,%r8,8), %xmm0             # xmm0 = mem[0],zero
	addsd	(%rsi,%r8,8), %xmm0
	movsd	%xmm0, (%rdx,%r8,8)
	movsd	8(%rdi,%r8,8), %xmm0            # xmm0 = mem[0],zero
	addsd	8(%rsi,%r8,8), %xmm0
	movsd	%xmm0, 8(%rdx,%r8,8)
	movsd	16(%rdi,%r8,8), %xmm0           # xmm0 = mem[0],zero
	addsd	16(%rsi,%r8,8), %xmm0
	movsd	%xmm0, 16(%rdx,%r8,8)
	movsd	24(%rdi,%r8,8), %xmm0           # xmm0 = mem[0],zero
	addsd	24(%rsi,%r8,8), %xmm0
	movsd	%xmm0, 24(%rdx,%r8,8)
	addq	$4, %r8
	cmpq	%r8, %rcx
	jne	.LBB1_12
	jmp	.LBB1_13
.LBB1_5:
	movabsq	$9223372036854775804, %rax      # imm = 0x7FFFFFFFFFFFFFFC
	andq	%rcx, %rax
	xorl	%r8d, %r8d
	.p2align	4
.LBB1_6:                                # =>This Inner Loop Header: Depth=1
	movupd	(%rdi,%r8,8), %xmm0
	movupd	16(%rdi,%r8,8), %xmm1
	movupd	(%rsi,%r8,8), %xmm2
	addpd	%xmm0, %xmm2
	movupd	16(%rsi,%r8,8), %xmm0
	addpd	%xmm1, %xmm0
	movupd	%xmm2, (%rdx,%r8,8)
	movupd	%xmm0, 16(%rdx,%r8,8)
	addq	$4, %r8
	cmpq	%r8, %rax
	jne	.LBB1_6
# %bb.7:
	cmpq	%rax, %rcx
	jne	.LBB1_8
.LBB1_13:
	movl	$_ZSt4cout, %edi
	movl	$.L.str, %esi
	movl	$20, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	_ZSt4cout(%rip), %rax
	movq	-24(%rax), %rax
	movq	_ZSt4cout+240(%rax), %rbx
	testq	%rbx, %rbx
	je	.LBB1_18
# %bb.14:
	cmpb	$0, 56(%rbx)
	je	.LBB1_16
# %bb.15:
	movzbl	67(%rbx), %eax
	jmp	.LBB1_17
.LBB1_16:
	movq	%rbx, %rdi
	callq	_ZNKSt5ctypeIcE13_M_widen_initEv
	movq	(%rbx), %rax
	movq	%rbx, %rdi
	movl	$10, %esi
	callq	*48(%rax)
.LBB1_17:
	movsbl	%al, %esi
	movl	$_ZSt4cout, %edi
	callq	_ZNSo3putEc
	movq	%rax, %rdi
	popq	%rbx
	.cfi_def_cfa_offset 8
	jmp	_ZNSo5flushEv                   # TAILCALL
.LBB1_18:
	.cfi_def_cfa_offset 16
	callq	_ZSt16__throw_bad_castv
.Lfunc_end1:
	.size	_Z10tensor_addIdEvPKT_S2_PS0_l, .Lfunc_end1-_Z10tensor_addIdEvPKT_S2_PS0_l
	.cfi_endproc
                                        # -- End function
	.text
	.globl	_Z35__device_stub__read_pattern_kernel2PKhmPh # -- Begin function _Z35__device_stub__read_pattern_kernel2PKhmPh
	.p2align	4
	.type	_Z35__device_stub__read_pattern_kernel2PKhmPh,@function
_Z35__device_stub__read_pattern_kernel2PKhmPh: # @_Z35__device_stub__read_pattern_kernel2PKhmPh
	.cfi_startproc
# %bb.0:
	subq	$104, %rsp
	.cfi_def_cfa_offset 112
	movq	%rdi, 72(%rsp)
	movq	%rsi, 64(%rsp)
	movq	%rdx, 56(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	leaq	80(%rsp), %r9
	movl	$_Z20read_pattern_kernel2PKhmPh, %edi
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$120, %rsp
	.cfi_adjust_cfa_offset -120
	retq
.Lfunc_end2:
	.size	_Z35__device_stub__read_pattern_kernel2PKhmPh, .Lfunc_end2-_Z35__device_stub__read_pattern_kernel2PKhmPh
	.cfi_endproc
                                        # -- End function
	.globl	_Z12read_patternPKhmPh          # -- Begin function _Z12read_patternPKhmPh
	.p2align	4
	.type	_Z12read_patternPKhmPh,@function
_Z12read_patternPKhmPh:                 # @_Z12read_patternPKhmPh
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$104, %rsp
	.cfi_def_cfa_offset 160
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rdx, %rbx
	movq	%rsi, %r14
	movq	%rdi, %r15
	movl	$_ZSt4cout, %edi
	movl	$.L.str.1, %esi
	movl	$11, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movq	%r14, %rsi
	callq	_ZNSo9_M_insertImEERSoT_
	movq	(%rax), %rcx
	movq	-24(%rcx), %rcx
	movq	240(%rax,%rcx), %r12
	testq	%r12, %r12
	je	.LBB3_9
# %bb.1:
	cmpb	$0, 56(%r12)
	je	.LBB3_3
# %bb.2:
	movzbl	67(%r12), %ecx
	jmp	.LBB3_4
.LBB3_3:
	movq	%r12, %rdi
	movq	%rax, %r13
	callq	_ZNKSt5ctypeIcE13_M_widen_initEv
	movq	(%r12), %rax
	movq	%r12, %rdi
	movl	$10, %esi
	callq	*48(%rax)
	movl	%eax, %ecx
	movq	%r13, %rax
.LBB3_4:
	movabsq	$4294967360, %r12               # imm = 0x100000040
	movsbl	%cl, %esi
	movq	%rax, %rdi
	callq	_ZNSo3putEc
	movq	%rax, %rdi
	callq	_ZNSo5flushEv
	movl	$10, %ebp
	leaq	192(%r12), %r13
	jmp	.LBB3_5
	.p2align	4
.LBB3_7:                                #   in Loop: Header=BB3_5 Depth=1
	decl	%ebp
	je	.LBB3_8
.LBB3_5:                                # =>This Inner Loop Header: Depth=1
	movq	%r13, %rdi
	movl	$1, %esi
	movq	%r12, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB3_7
# %bb.6:                                #   in Loop: Header=BB3_5 Depth=1
	movq	%r15, 72(%rsp)
	movq	%r14, 64(%rsp)
	movq	%rbx, 56(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movl	$_Z20read_pattern_kernel2PKhmPh, %edi
	leaq	80(%rsp), %r9
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	jmp	.LBB3_7
.LBB3_8:
	addq	$104, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB3_9:
	.cfi_def_cfa_offset 160
	callq	_ZSt16__throw_bad_castv
.Lfunc_end3:
	.size	_Z12read_patternPKhmPh, .Lfunc_end3-_Z12read_patternPKhmPh
	.cfi_endproc
                                        # -- End function
	.p2align	4                               # -- Begin function __hip_module_ctor
	.type	__hip_module_ctor,@function
__hip_module_ctor:                      # @__hip_module_ctor
	.cfi_startproc
# %bb.0:
	subq	$40, %rsp
	.cfi_def_cfa_offset 48
	movq	__hip_gpubin_handle_6f79350ec0a424b5(%rip), %rdi
	testq	%rdi, %rdi
	jne	.LBB4_2
# %bb.1:
	movl	$__hip_fatbin_wrapper, %edi
	callq	__hipRegisterFatBinary
	movq	%rax, %rdi
	movq	%rax, __hip_gpubin_handle_6f79350ec0a424b5(%rip)
.LBB4_2:
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z20read_pattern_kernel2PKhmPh, %esi
	movl	$.L__unnamed_1, %edx
	movl	$.L__unnamed_1, %ecx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	movl	$__hip_module_dtor, %edi
	addq	$40, %rsp
	.cfi_def_cfa_offset 8
	jmp	atexit                          # TAILCALL
.Lfunc_end4:
	.size	__hip_module_ctor, .Lfunc_end4-__hip_module_ctor
	.cfi_endproc
                                        # -- End function
	.p2align	4                               # -- Begin function __hip_module_dtor
	.type	__hip_module_dtor,@function
__hip_module_dtor:                      # @__hip_module_dtor
	.cfi_startproc
# %bb.0:
	movq	__hip_gpubin_handle_6f79350ec0a424b5(%rip), %rdi
	testq	%rdi, %rdi
	je	.LBB5_2
# %bb.1:
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	__hipUnregisterFatBinary
	movq	$0, __hip_gpubin_handle_6f79350ec0a424b5(%rip)
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
.LBB5_2:
	retq
.Lfunc_end5:
	.size	__hip_module_dtor, .Lfunc_end5-__hip_module_dtor
	.cfi_endproc
                                        # -- End function
	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"hahah: in tensor_add"
	.size	.L.str, 21

	.type	_Z20read_pattern_kernel2PKhmPh,@object # @_Z20read_pattern_kernel2PKhmPh
	.section	.rodata,"a",@progbits
	.globl	_Z20read_pattern_kernel2PKhmPh
	.p2align	3, 0x0
_Z20read_pattern_kernel2PKhmPh:
	.quad	_Z35__device_stub__read_pattern_kernel2PKhmPh
	.size	_Z20read_pattern_kernel2PKhmPh, 8

	.type	.L.str.1,@object                # @.str.1
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.1:
	.asciz	"x_stride = "
	.size	.L.str.1, 12

	.type	.L__unnamed_1,@object           # @0
.L__unnamed_1:
	.asciz	"_Z20read_pattern_kernel2PKhmPh"
	.size	.L__unnamed_1, 31

	.type	__hip_fatbin_wrapper,@object    # @__hip_fatbin_wrapper
	.section	.hipFatBinSegment,"a",@progbits
	.p2align	3, 0x0
__hip_fatbin_wrapper:
	.long	1212764230                      # 0x48495046
	.long	1                               # 0x1
	.quad	__hip_fatbin_6f79350ec0a424b5
	.quad	0
	.size	__hip_fatbin_wrapper, 24

	.type	__hip_gpubin_handle_6f79350ec0a424b5,@object # @__hip_gpubin_handle_6f79350ec0a424b5
	.local	__hip_gpubin_handle_6f79350ec0a424b5
	.comm	__hip_gpubin_handle_6f79350ec0a424b5,8,8
	.section	.init_array,"aw",@init_array
	.p2align	3, 0x0
	.quad	__hip_module_ctor
	.type	__hip_cuid_6f79350ec0a424b5,@object # @__hip_cuid_6f79350ec0a424b5
	.bss
	.globl	__hip_cuid_6f79350ec0a424b5
__hip_cuid_6f79350ec0a424b5:
	.byte	0                               # 0x0
	.size	__hip_cuid_6f79350ec0a424b5, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.3 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z35__device_stub__read_pattern_kernel2PKhmPh
	.addrsig_sym __hip_module_ctor
	.addrsig_sym __hip_module_dtor
	.addrsig_sym _ZSt4cout
	.addrsig_sym _Z20read_pattern_kernel2PKhmPh
	.addrsig_sym __hip_fatbin_6f79350ec0a424b5
	.addrsig_sym __hip_fatbin_wrapper
	.addrsig_sym __hip_cuid_6f79350ec0a424b5

# __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu-
