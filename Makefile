# Makefile for CUDA transpose program

NVCC := nvcc
CFLAGS := -O3 -arch=sm_60
TARGET := programa
SRC := main.cu

.PHONY: all build run clean

all: build

build: $(SRC)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRC)

run: build
	./$(TARGET) 1024 1024

clean:
	rm -f $(TARGET) 