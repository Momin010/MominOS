# MominOS Pro-X 

MominOS Pro-X is a high-density, 64-bit hobbyist operating system designed for x86_64 architecture. It transitions from 16-bit Real Mode to 64-bit Long Mode to provide a stable, extensible environment.

##  Architecture
- **Stage 1 MBR**: Primary boot sector (512 bytes).
- **Stage 2 Loader**: Handles A20, GDT, 4-level Paging, and the jump to Long Mode.
- **Micro-Kernel**: 64-bit core with integrated VGA, Keyboard, and IDT subsystems.
- **Interactive Shell**: Functional CLI for hardware interaction.

##  Features
- **Long Mode Execution**: Pure 64-bit kernel logic.
- **Scrolling VGA Driver**: High-density display management.
- **Exception Handling**: Professional IDT for all 32 CPU exceptions.
- **Modular Drivers**: Separated VGA and Keyboard logic.

##  Build & Run
### Prerequisites
- [NASM](https://www.nasm.us/)
- [QEMU](https://www.qemu.org/) (for emulation)

### Building
```powershell
.\scripts\build.bat
```

### Running
```powershell
qemu-system-x86_64 -fda bin/mominos.img
```

##  Structure
- `src/boot/`: Bootloader stages.
- `src/kernel/`: Core kernel and interrupt handling.
- `src/drivers/`: Hardware interaction layers.
- `scripts/`: Build and utility scripts.

##  License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
