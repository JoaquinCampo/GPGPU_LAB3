NVCC := nvcc
CFLAGS := -O3 -arch=compute_50
LDFLAGS := -ldl
TARGETS := ej1 ej2_part1 ej2_part2
SRCS := ej1.cu ej2_part1.cu ej2_part2.cu

.PHONY: all build run clean

all: build

build: $(SRCS)
	$(NVCC) $(CFLAGS) -o ej1 ej1.cu
	$(NVCC) $(CFLAGS) -o ej2_part1 ej2_part1.cu
	$(NVCC) $(CFLAGS) -o ej2_part2 ej2_part2.cu
