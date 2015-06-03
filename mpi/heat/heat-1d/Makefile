heat := heat-1d
heat_objs := heat-1d.o

CC := mpicc
CFLAGS := -std=c99 -Wall -O2
LDFLAGS := -lm

.PHONY: all clean

all: $(heat)

$(heat): $(heat_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

heat-1d.o: heat-1d.c

clean:
	@rm -rf *.o $(heat)
