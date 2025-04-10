# 编译器设置
CC = riscv64-unknown-linux-musl-gcc
CFLAGS = -I./include -Wall -Wextra
LDFLAGS = -L./lib -Wl,--dynamic-linker=/lib/ld-musl-riscv64v_xthead.so.1
LDLIBS = -lcurl -lcjson -lssl -lcrypto -lz


# 目标文件
TARGET = deepseek
SRC = sample_ds.c
OBJ = $(SRC:.c=.o)

# 默认目标
all: $(TARGET)

# 链接目标
$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# 编译规则
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# 清理
clean:
	rm -f $(TARGET) $(OBJ)

# 声明伪目标
.PHONY: all clean
