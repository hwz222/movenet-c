CC = gcc
CFLAGS = -Iinclude -Wall -g
SRC_DIR = src
OBJ_DIR = build
TARGET = $(OBJ_DIR)/circle_tool

# 自動搜尋 src 底下所有的 .c 檔案
SRCS = $(wildcard $(SRC_DIR)/*.c)
# 將 src/*.c 轉換成 build/*.o
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(OBJ_DIR)