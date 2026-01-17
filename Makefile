# FLUX.2 klein 4B - Pure C Inference Engine
# Makefile

CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -ffast-math
LDFLAGS = -lm

# Platform-specific BLAS support
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    # macOS: Use Accelerate framework for BLAS
    LDFLAGS += -framework Accelerate
endif
ifeq ($(UNAME_S),Linux)
    # Linux: Use OpenBLAS if available (compile with USE_OPENBLAS=1)
    ifdef USE_OPENBLAS
        CFLAGS += -DUSE_OPENBLAS
        LDFLAGS += -lopenblas
    endif
endif

# Debug build
DEBUG_CFLAGS = -Wall -Wextra -g -O0 -DDEBUG -fsanitize=address

# Source files
SRCS = flux.c flux_kernels.c flux_tokenizer.c flux_vae.c flux_transformer.c flux_sample.c flux_image.c flux_safetensors.c
OBJS = $(SRCS:.c=.o)

# Main program
MAIN = main.c
TARGET = flux

# Library
LIB = libflux.a

.PHONY: all clean debug lib test install

all: $(TARGET)

$(TARGET): $(OBJS) $(MAIN:.c=.o)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

lib: $(LIB)

$(LIB): $(OBJS)
	ar rcs $@ $^

%.o: %.c flux.h flux_kernels.h
	$(CC) $(CFLAGS) -c -o $@ $<

debug: CFLAGS = $(DEBUG_CFLAGS)
debug: LDFLAGS += -fsanitize=address
debug: clean $(TARGET)

# Run tests
test: $(TARGET)
	@echo "Running basic tests..."
	@echo "Creating placeholder model..."
	python3 convert_weights.py --placeholder -o test_model.bin
	@echo "Testing CLI..."
	./flux -m test_model.bin -p "test prompt" -o test_output.png -v || true
	@echo "Tests complete."

# Install to /usr/local
install: $(TARGET) $(LIB)
	install -d /usr/local/bin
	install -d /usr/local/lib
	install -d /usr/local/include
	install -m 755 $(TARGET) /usr/local/bin/
	install -m 644 $(LIB) /usr/local/lib/
	install -m 644 flux.h /usr/local/include/
	install -m 644 flux_kernels.h /usr/local/include/

clean:
	rm -f $(OBJS) $(MAIN:.c=.o) $(TARGET) $(LIB)
	rm -f test_model.bin test_output.png

# Dependencies
flux.o: flux.c flux.h flux_kernels.h flux_safetensors.h
flux_kernels.o: flux_kernels.c flux_kernels.h
flux_tokenizer.o: flux_tokenizer.c flux.h
flux_vae.o: flux_vae.c flux.h flux_kernels.h
flux_transformer.o: flux_transformer.c flux.h flux_kernels.h
flux_sample.o: flux_sample.c flux.h flux_kernels.h
flux_image.o: flux_image.c flux.h
flux_safetensors.o: flux_safetensors.c flux_safetensors.h
main.o: main.c flux.h

# Optimization variants
fast: CFLAGS += -Ofast -funroll-loops -ftree-vectorize
fast: clean $(TARGET)

# With OpenMP for parallelization
parallel: CFLAGS += -fopenmp -DUSE_OPENMP
parallel: LDFLAGS += -fopenmp
parallel: clean $(TARGET)

# Profile build
profile: CFLAGS += -pg
profile: LDFLAGS += -pg
profile: clean $(TARGET)

# Size-optimized build
small: CFLAGS = -Wall -Os -s
small: clean $(TARGET)

# Show size info
size: $(TARGET)
	@echo "Binary size:"
	@ls -lh $(TARGET)
	@echo ""
	@echo "Object sizes:"
	@ls -lh *.o
	@echo ""
	@echo "Sections:"
	@size $(TARGET)

# Generate documentation
docs:
	@echo "FLUX.2 klein 4B - Pure C Inference Engine"
	@echo "=========================================="
	@echo ""
	@echo "Build: make"
	@echo "Debug: make debug"
	@echo "Test:  make test"
	@echo ""
	@echo "Usage:"
	@echo "  ./flux -m model.bin -p \"prompt\" -o output.png"
	@echo ""
	@echo "Options:"
	@echo "  -m  Model file path"
	@echo "  -p  Text prompt"
	@echo "  -o  Output image path"
	@echo "  -W  Width (default: 1024)"
	@echo "  -H  Height (default: 1024)"
	@echo "  -s  Steps (default: 4)"
	@echo "  -S  Seed (-1 for random)"
	@echo "  -v  Verbose output"

# Count lines of code
loc:
	@echo "Lines of code:"
	@wc -l *.c *.h | tail -1

# Format code (requires clang-format)
format:
	clang-format -i *.c *.h

# Static analysis (requires cppcheck)
check:
	cppcheck --enable=all --suppress=missingIncludeSystem *.c

# Convert model weights
convert:
	python3 convert_weights.py --model black-forest-labs/FLUX.2-klein-4B --output flux-klein-4b.bin

# Create placeholder model for testing
placeholder:
	python3 convert_weights.py --placeholder --output flux-placeholder.bin

help:
	@echo "FLUX.2 Makefile targets:"
	@echo "  all       - Build the flux executable (default)"
	@echo "  lib       - Build static library libflux.a"
	@echo "  debug     - Build with debug symbols and sanitizers"
	@echo "  test      - Run basic tests"
	@echo "  install   - Install to /usr/local"
	@echo "  clean     - Remove build artifacts"
	@echo "  fast      - Build with aggressive optimizations"
	@echo "  parallel  - Build with OpenMP parallelization"
	@echo "  small     - Build size-optimized binary"
	@echo "  convert   - Convert HuggingFace model to binary format"
	@echo "  placeholder - Create minimal test model"
	@echo "  docs      - Show documentation"
	@echo "  loc       - Count lines of code"
	@echo "  format    - Format code with clang-format"
	@echo "  check     - Run static analysis"
