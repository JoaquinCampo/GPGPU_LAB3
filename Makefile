NVCC := nvcc
CFLAGS := -O3 -arch=compute_50
TARGET := ej1_part2
SRC := ej1_part2.cu

.PHONY: all build run clean

all: build

build: $(SRC)
        $(NVCC) $(CFLAGS) -o $(TARGET) $(SRC)

run: build
        ./$(TARGET) 1024 1024

clean:
        rm -f $(TARGET)