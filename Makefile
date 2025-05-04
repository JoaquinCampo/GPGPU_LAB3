NVCC := nvcc
CFLAGS := -O3 -arch=compute_50
LDFLAGS := -ldl

TARGET := ej1_part1
SRC := ej1_part1.cu

.PHONY: all build run clean

all: build

build: $(SRC)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRC) 

run: build
	./$(TARGET) 1024 1024





