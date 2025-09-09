# Makefile for Convex Partition C Extensions
CC = gcc
CFLAGS = -Wall -Wextra -O2 -fPIC -std=c99
LIBS = -lm

# Source files
SOURCES = convex_partition.c
OBJECTS = $(SOURCES:.c=.o)
HEADERS = convex_partition.h

# Targets
TARGET_LIB = libconvex_partition.so
TARGET_TEST = test_convex_partition

.PHONY: all clean test lib

all: lib $(TARGET_TEST)

# Shared library for Python integration
lib: $(TARGET_LIB)

$(TARGET_LIB): $(OBJECTS)
	$(CC) -shared -o $@ $^ $(LIBS)

# Test executable
$(TARGET_TEST): test_main.c $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

# Object files
%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

# Test target
test: $(TARGET_TEST)
	./$(TARGET_TEST)
	@echo "✅ C tests passed"

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET_LIB) $(TARGET_TEST)
	@echo "🧹 Cleaned build artifacts"

# Debug info
debug:
	@echo "CC: $(CC)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "SOURCES: $(SOURCES)"
	@echo "OBJECTS: $(OBJECTS)"
	@echo "TARGET_LIB: $(TARGET_LIB)"
	@echo "TARGET_TEST: $(TARGET_TEST)"

# Install (copy shared library to appropriate location)
install: $(TARGET_LIB)
	mkdir -p ../../convex_partition/lib
	cp $(TARGET_LIB) ../../convex_partition/lib/
	@echo "📦 Installed shared library"