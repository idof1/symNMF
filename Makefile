CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors

PYTHON_INCLUDE = $(shell python3-config --includes)

# Target executable
TARGET = symnmf

# Object files
OBJS = symnmf.o

# Default target
all: $(TARGET)

# Build the symnmf executable
$(TARGET): $(OBJS)
	$(CC) -o $(TARGET) $(OBJS) $(CFLAGS) -lm

# Compile symnmf.c with strict ANSI C flags
symnmf.o: symnmf.c
	$(CC) -c symnmf.c $(CFLAGS)

# Clean up generated files
clean:
	rm -f $(OBJS) $(TARGET)
