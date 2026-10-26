"""TPU SoC co-sim: build firmware, compile tb_tpu_soc with iverilog, run.

PASS 判定：C 程序在 UART 打印 TPU_SOC_PASS（TB 监听并输出 "RESULT: PASS"）。
用法：python sim/tpu_soc/run.py
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
IV = Path("D:/vsc/msys64/ucrt64/bin")
BUILD = HERE / "build"

# iverilog/vvp 是 MSYS2 程序，依赖 ucrt64/bin 下的 DLL（如 libgcc_s_seh-1.dll）。
# Windows 原生 subprocess 不会继承 MSYS2 的 PATH，缺了会报 STATUS_DLL_NOT_FOUND
# （returncode = 0xC0000135）且无任何输出。这里把 bin 目录显式加进 PATH。
_ENV = dict(os.environ)
_ENV["PATH"] = str(IV) + os.pathsep + _ENV.get("PATH", "")

NAMES = ("alu br_alu l_alu regfile csr_file mycpu_sync "
         "inst_bus_interconnect data_bus_interconnect "
         "inst_bram_adapter data_bram_adapter simple_mmio "
         "dual_sram_to_pango_ddr_bridge uart_tx uart_rx uart_rx_fifo "
         "uart_mmio soc_top preprocess_ddr_read_arbiter "
         "tpu_mmio tpu_dma tpu_conv_engine").split()
SOURCES = ([ROOT / "myriscv" / (n + ".v") for n in NAMES]
           + [ROOT / "sim/uart_echo/tb/echo_bram.v",
              ROOT / "sim/br_ex_redirect/mul_behavioral.v",
              HERE / "tb_tpu_soc.v"])
DAT = ROOT / "tests/tpu_conv/build/program.dat"


def main():
    BUILD.mkdir(exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "tests/tpu_conv/build.py")],
                   check=True)
    exe = BUILD / "tb.vvp"
    defines = ["-DTPU_SOC_DEBUG"] if "--debug" in sys.argv else []
    r = subprocess.run([str(IV / "iverilog.exe"), "-g2012",
                        "-I", str(ROOT / "myriscv"),
                        *defines,
                        "-s", "tb_tpu_soc", "-o", str(exe),
                        *map(str, SOURCES)],
                       capture_output=True, text=True, timeout=180, env=_ENV)
    (BUILD / "compile.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    if r.returncode:
        print(r.stdout + r.stderr)
        raise SystemExit("COMPILE FAIL")
    t0 = time.monotonic()
    r = subprocess.run([str(IV / "vvp.exe"), str(exe),
                        f"+DAT={DAT}", f"+ROM={DAT}"],
                       capture_output=True, text=True, timeout=900, env=_ENV)
    text = r.stdout + r.stderr
    (BUILD / "sim.log").write_text(text, encoding="utf-8")
    seconds = round(time.monotonic() - t0, 2)
    print(text[-4000:])
    ok = "RESULT: PASS" in text
    print(f"==== tpu_soc {'PASS' if ok else 'FAIL'} ({seconds}s) ====")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
