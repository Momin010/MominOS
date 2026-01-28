@echo off
setlocal
set BIN_DIR=bin
set SRC_DIR=src

if not exist %BIN_DIR% mkdir %BIN_DIR%

echo [BUILD] Assembling Stage 1 MBR...
nasm -f bin %SRC_DIR%\boot\boot_mbr.asm -o %BIN_DIR%\boot_mbr.bin

echo [BUILD] Assembling Stage 2 Loader...
nasm -f bin %SRC_DIR%\boot\boot_loader.asm -o %BIN_DIR%\boot_loader.bin

echo [BUILD] Assembling 64-bit Kernel...
nasm -f bin %SRC_DIR%\kernel\kernel64.asm -i %SRC_DIR%\kernel\ -o %BIN_DIR%\kernel64.bin

echo [BUILD] Creating Bootable Image...
copy /b %BIN_DIR%\boot_mbr.bin + %BIN_DIR%\boot_loader.bin + %BIN_DIR%\kernel64.bin %BIN_DIR%\mominos.img

echo [DONE] Build complete. Artifact: %BIN_DIR%\mominos.img
endlocal
