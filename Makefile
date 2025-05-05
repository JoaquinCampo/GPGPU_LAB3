NVCC := nvcc
CFLAGS := -O3 -arch=compute_50
LDFLAGS := -ldl
TARGETS := ej1_part1 ej1_part2 ej2_part1 ej2_part2
SRCS := ej1_part1.cu ej1_part2.cu ej2_part1.cu ej2_part2.cu

.PHONY: all build run clean

all: build

build: $(SRCS)
	$(NVCC) $(CFLAGS) -o ej1_part1 ej1_part1.cu
	$(NVCC) $(CFLAGS) -o ej1_part2 ej1_part2.cu
	$(NVCC) $(CFLAGS) -o ej2_part1 ej2_part1.cu
	$(NVCC) $(CFLAGS) -o ej2_part2 ej2_part2.cu

run: build
	./ej1_part1 1024 1024
	./ej1_part2 1024 1024
	./ej2_part1 1024 1024
	./ej2_part2 1024 1024