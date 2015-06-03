heat := heat
heat_objs := heat.o

CC := gcc
CFLAGS := -std=c99 -Wall -O2
LDFLAGS := -lm

.PHONY: all clean

all: $(heat)

$(heat): $(heat_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

heat.o: heat.c

clean:
	@rm -rf *.o $(heat)
