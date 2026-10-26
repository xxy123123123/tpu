    .section .text.startup,"ax",@progbits
    .globl _start
_start:
    .word 0x30047073   # csrwi mstatus,0 关全局中断
    .word 0x30401073   # csrwi mie,0
    .option push
    .option norelax
    la gp,__global_pointer$
    .option pop
    la sp,__stack_top
    la t0,trap_spin
    .word 0x30529073   # csrw mtvec,t0
    # 清 BSS（.data 已由镜像按 VMA 直接装载，LMA==VMA 无需拷贝）
    la t0,__bss_start
    la t1,__bss_end
1:  bgeu t0,t1,2f
    sw zero,0(t0)
    addi t0,t0,4
    j 1b
2:  call main
3:  j 3b
    .globl trap_spin
trap_spin:
    j trap_spin
