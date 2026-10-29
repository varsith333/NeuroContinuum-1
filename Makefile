# NeuroContinuum-1 — top-level orchestration
#
#   make            # everything (test + demo + bench + rtl)
#   make test       # verify the four novel blocks + package import
#   make demo       # flagship continual-learning demo (bridge/dam/factory)
#   make frozen     # same demo with learning OFF (the A/B baseline)
#   make bench      # architecture + energy benchmark vs CPU/GPU
#   make rtl        # RTL simulation (iverilog, or `make -C rtl xsim`)
#                    #   Windows without make: powershell -ExecutionPolicy Bypass
#                    #                        -File run_all.ps1   (whole project)
#   make clean      # remove generated results/logs
#
# PowerShell note: on Windows run the commands after the tabs manually with
# `;` instead of chaining with `&&`, or use `make -f Makefile` from Git Bash.

PY ?= python

.PHONY: all test demo frozen bench rtl results clean

all: test demo bench

test:
	$(PY) -m simulator.examples.test_novel_features

demo:
	$(PY) -m simulator.examples.demo_continual --n 400

frozen:
	$(PY) -m simulator.examples.demo_continual --n 400 --no-learn

bench:
	$(PY) -m architecture.benchmark

rtl:
	$(MAKE) -C rtl

# regenerate every published number in one pass
results: test demo frozen bench rtl
	@echo "-> reports written to architecture/results/ and rtl/work/"

clean:
	$(MAKE) -C rtl clean
	rm -f architecture/results/*.json architecture/results/*.md
