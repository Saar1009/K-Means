# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -Werror -ansi -pedantic-errors

# Target executable and object files
TARGET = symnmf
OBJS   = SymNMF.o

# Build the target
$(TARGET): $(OBJS)
	$(CC) -o $(TARGET) $(OBJS) $(CFLAGS) -lm

# Build the object file
SymNMF.o: SymNMF.c
	$(CC) -c SymNMF.c $(CFLAGS) -o SymNMF.o

# Clean up
clean:
	rm -f $(TARGET) $(OBJS)