// TPU SoC 级软硬件协同测试。
// 场景与 sim/tpu/tb_tpu_wdma.v 完全一致（可手算）：
//   输入 4x4x8 全 1，3x3 卷积 pad=1 stride=1，cout=32（npg=2，走权重乒乓）
//   权重 co 偶数=1 / 奇数=2，bias 全 0，shift=0 relu=0
//   输出值 = 有效 tap 数 x 8 x w(co)：角 4taps / 边 6taps / 内 9taps
//   期望直方图：32->64, 64->64, 48->128, 96->128, 72->64, 144->64, 其它->0
//
// 数据通路：CPU 经 0x4201_xxxx 通用 DDR 区（避开 preprocess 只读输出窗口，
// 桥平移后 = AXI 扁平 0x0201_xxxx）准备激活/权重；TPU DMA 以扁平地址直接读。
// store miss 不分配缓存行，且 DMA 完成前 CPU 不预读输出区，随后读回不会命中陈旧行。
#include <stdint.h>

#define UART_DATA   (*(volatile uint8_t  *)0x10001000u)
#define UART_STATUS (*(volatile uint32_t *)0x10001008u)

#define TPU_BASE 0x10005000u
#define TPU_W(off, v) (*(volatile uint32_t *)(TPU_BASE + (off)) = (v))
#define TPU_R(off)    (*(volatile uint32_t *)(TPU_BASE + (off)))

/* CPU 侧通用 DDR 地址（interconnect 平移 0x4000_0000 后 = 0x0201_xxxx） */
#define ACT_CPU ((volatile uint8_t *)0x42010000u)
#define WT_CPU  ((volatile uint8_t *)0x42011000u)
#define OUT_CPU ((volatile uint8_t *)0x42012000u)
/* TPU DMA 侧扁平 AXI 地址 */
#define ACT_AXI 0x02010000u
#define WT_AXI  0x02011000u
#define OUT_AXI 0x02012000u

#define H 4
#define W 4
#define CIN 8u
#define COUT 32u
#define ACT_BYTES (H * W * CIN)   /* 128 */
#define OUT_BYTES (H * W * COUT)  /* 512 */

static void putc_(char c) { while (!(UART_STATUS & 1u)) {} UART_DATA = (uint8_t)c; }
static void puts_(const char *s) { while (*s) putc_(*s++); putc_('\n'); }
static void puthex(uint32_t v) {
    int i;
    for (i = 7; i >= 0; i--) {
        uint32_t d = (v >> (i * 4)) & 0xfu;
        putc_((char)(d < 10 ? '0' + d : 'a' + d - 10));
    }
}
static void putdec(uint32_t v) {
    char buf[10];
    int n = 0;
    if (!v) { putc_('0'); return; }
    while (v) { buf[n++] = (char)('0' + v % 10u); v /= 10u; }
    while (n) putc_(buf[--n]);
}
static void kv_u32(const char *k, uint32_t v) { puts_(k); puthex(v); }

int main(void) {
    uint32_t i, co, kk, st, spins;
    int errors = 0;
    uint32_t c32 = 0, c64 = 0, c48 = 0, c96 = 0, c72 = 0, c144 = 0, cother = 0;

    puts_("[tpu] boot");

    /* [1] ID */
    if (TPU_R(0x00) != 0x54505531u) { puts_("[tpu] FAIL id"); errors++; }
    else puts_("[tpu] id OK (TPU1)");

    /* [2] 准备数据（CPU 写非缓存窗口 -> 直达 DDR 模型） */
    for (i = 0; i < ACT_BYTES; i++) ACT_CPU[i] = 1u;
    for (co = 0; co < COUT; co++)
        for (kk = 0; kk < 9u; kk++)
            for (i = 0; i < CIN; i++)
                WT_CPU[co * 72u + kk * CIN + i] = (co & 1u) ? 2u : 1u;
    for (i = 0; i < OUT_BYTES; i++) OUT_CPU[i] = 0xEEu;   /* 毒化，DMA 应覆盖 */
    puts_("[tpu] buffers ready");

    /* [3] 层配置：h=4 w=4 cin=8 cout=32 stride=1 pad=1 k=3 relu=0 shift=0 */
    TPU_W(0x0C, ((uint32_t)H << 16) | (uint32_t)W);
    TPU_W(0x10, (CIN << 16) | COUT);
    TPU_W(0x14, 5u);

    /* [4] DMA 配置（激活+输出走 DMA，权重也走 DMA） */
    TPU_W(0x40, ACT_AXI);
    TPU_W(0x44, ACT_BYTES);
    TPU_W(0x48, OUT_AXI);
    TPU_W(0x4C, OUT_BYTES);
    TPU_W(0x50, 1u);          /* dma_en */
    TPU_W(0x58, WT_AXI);
    TPU_W(0x5C, 1u);          /* wt_en */
    if ((TPU_R(0x5C) & 1u) != 1u) { puts_("[tpu] FAIL wt_en"); errors++; }

    /* [5] bias = 0（start 前写，S_IDLE 时 b_ready 恒 1） */
    for (co = 0; co < COUT; co++) {
        while (TPU_R(0x08) & 8u) {}     /* bit3 = b_pend */
        TPU_W(0x24, co);
        TPU_W(0x20, 0u);
    }

    /* [6] start */
    TPU_W(0x04, 1u);

    /* [7] 轮询 done（bit1），带超时 */
    spins = 0;
    do { st = TPU_R(0x08); spins++; } while (!(st & 2u) && spins < 200000u);
    if (!(st & 2u)) { puts_("[tpu] FAIL timeout"); errors++; }
    if (st & 0x40u) { puts_("[tpu] FAIL error bit"); errors++; }
    if (TPU_R(0x38) != 0u) { puts_("[tpu] FAIL err reg"); errors++; }

    /* [8] 性能计数器（诊断打印，不做断言） */
    kv_u32("[tpu] cyc   =", TPU_R(0x34));
    kv_u32("[tpu] pf_tot=", TPU_R(0x6C));
    kv_u32("[tpu] pf_cmp=", TPU_R(0x70));
    kv_u32("[tpu] pf_act=", TPU_R(0x74));
    kv_u32("[tpu] pf_stv=", TPU_R(0x78));
    kv_u32("[tpu] pf_out=", TPU_R(0x7C));
    kv_u32("[tpu] pf_wld=", TPU_R(0x80));
    kv_u32("[tpu] pf_drn=", TPU_R(0x84));

    /* [9] 校验输出直方图（CPU 经非缓存别名读回） */
    for (i = 0; i < OUT_BYTES; i++) {
        switch (OUT_CPU[i]) {
        case 32u:  c32++;  break;
        case 64u:  c64++;  break;
        case 48u:  c48++;  break;
        case 96u:  c96++;  break;
        case 72u:  c72++;  break;
        case 144u: c144++; break;
        default:   cother++; break;
        }
    }
    kv_u32("[tpu] hist 32  =", c32);
    kv_u32("[tpu] hist 64  =", c64);
    kv_u32("[tpu] hist 48  =", c48);
    kv_u32("[tpu] hist 96  =", c96);
    kv_u32("[tpu] hist 72  =", c72);
    kv_u32("[tpu] hist 144 =", c144);
    kv_u32("[tpu] hist oth =", cother);
    if (c32 != 64u || c64 != 64u || c48 != 128u || c96 != 128u ||
        c72 != 64u || c144 != 64u || cother != 0u) {
        puts_("[tpu] FAIL histogram");
        errors++;
    }

    if (errors) { puts_("TPU_SOC_FAIL"); putdec((uint32_t)errors); }
    else          puts_("TPU_SOC_PASS");
    return errors;
}
