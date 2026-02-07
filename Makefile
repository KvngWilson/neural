CC = gcc
CFLAGS = -Wall -Wextra -O2
LDFLAGS = -lm
TARGET = neural
OBJS = main.o neural.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

main.o: main.c neural.h
	$(CC) $(CFLAGS) -c main.c

neural.o: neural.c neural.h
	$(CC) $(CFLAGS) -c neural.c

clean:
	rm -f $(OBJS) $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
