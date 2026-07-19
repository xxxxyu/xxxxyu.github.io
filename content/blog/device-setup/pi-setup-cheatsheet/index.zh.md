+++
title = "Pi 设备配置速查表"
date = "2025-01-03"
updated = "2026-07-11"
description = "适用于 Raspberry Pi 及类似单板计算机的实用配置清单。"
path = "zh/blog/pi-setup-cheatsheet"
weight = 10

[extra]
series = "blog/device-setup/_index.md"
ai_translation_source = "en"

[taxonomies]
tags = ["Development", "Linux", "Edge Device"]
+++

这份速查表是根据我自己的开发需求整理的，主要适用于 Orange Pi、Raspberry Pi 等开发板。

## 硬件

### 网络连接

有线或无线网络均可。

### 键盘、鼠标和显示器

用于首次开机配置和本地调试。

### 已写入操作系统的 TF（MicroSD）卡

- 常规开发推荐使用最新的 Ubuntu LTS
  - 我更倾向于使用服务器版系统，因为它比桌面镜像更轻量
  - 同时也能避免 GUI 干扰性能测量
- 预算允许的话，建议使用容量较大的存储卡（我使用的是 ≥256 GB）
  - 默认的 32/64 GB 对许多任务而言并不够用
- 镜像写入工具：[BalenaEtcher](https://etcher.balena.io/#download-etcher)（通用）和 [Raspberry Pi Imager](https://www.raspberrypi.com/software/)（Raspberry Pi）
  - Raspberry Pi Imager 支持通过自定义配置完成无头安装

## 系统

### 按需设置或修改用户名和密码

- 推荐格式：设备名（`raspi`、`orangepi`）[+ 用户信息，适用于设备不共用的情况]

### 连接（无线）网络

- Raspberry Pi 在无头安装时可以自动连接网络
  - 默认由 `cloud-init`、`netplan` 和 `networkd` 管理网络
  - 我改用了自己更熟悉的 `NetworkManager`
- 其他运行 Ubuntu 的开发板可以使用 `NetworkManager`
  - 扫描可用网络：`nmcli dev wifi`
  - 通过 SSID 连接：`nmcli dev wifi connect <ssid> password <passwd>`

### 按需配置网络

- 清华大学 TUNET 认证：[GoAuthing](https://github.com/z4yx/GoAuthing/)
  - 如果代理导致无法访问 GitHub，也可以从 [TUNA](https://mirrors.tuna.tsinghua.edu.cn/github-release/z4yx/GoAuthing/LatestRelease/) 获取发布版本
  - 预编译的 Linux ARM64 可执行文件：[auth-thu.linux.arm64](https://mirrors.tuna.tsinghua.edu.cn/github-release/z4yx/GoAuthing/LatestRelease/auth-thu.linux.arm64)
- 配置代理：TODO
  - 使用公共网络时，不要允许来自局域网的连接

### 安装或更新必要的软件包

- 必要时更换软件源
- 首先更新软件包列表并升级：`sudo apt update && sudo apt upgrade`

### 配置 SSH 服务以便远程访问

- 使用 `sshd` 启动 SSH 服务
- 使用 `ifconfig` 查看设备 IP
- 使用 `ssh -p <port> user@ip` 测试登录
- 使用 `ssh-copy-id` 配置密钥认证

### 其他

（可选）使用厂商提供的工具进行更直观的交互式配置，例如 Raspberry Pi 的 `raspi-config` 和 Orange Pi 的 `orangepi-config`。

## 开发环境

### 基础工具和软件包

- tmux：`sudo apt install tmux`
- git：`sudo apt install git`
  - 将 SSH 公钥添加到 GitHub，以便通过 SSH 克隆仓库
  - 提交代码前配置用户名和邮箱
- code-server（适用于平台不支持 VS Code Remote 的情况）

### 构建工具和编译器

- CMake（大多数项目要求 >=3.22）
  - 通过 apt 安装：`sudo apt install cmake`
  - 从官方网站安装：[下载页面](https://cmake.org/download/)
  - 更新至最新版：[更新 CMake](#geng-xin-cmake)
- GCC
  - 通过 apt 安装：`sudo apt install build-essential`
- LLVM + Clang
  - 从官方网站安装：[下载页面](https://apt.llvm.org/)
  - 一行命令安装最新版：`bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"`
- 注意：如果系统中同时装有多个编译器（GCC、Clang 或同一编译器的不同版本），请确认环境变量 `$CC` 和 `$CXX` 设置正确，否则 CMake 等构建工具可能会选错编译器。

### Python 环境

- Miniconda
  - 大多数情况下的首选方案
  - 一行命令安装（注意操作系统和架构）
    - `curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh`
    - `bash ~/Miniconda3-latest-Linux-aarch64.sh`
      - 不要使用 `sudo`，否则 Miniconda 会安装到其他位置
    - 阅读协议后按 Enter，并输入 yes
- Virtualenv
  - 在项目有要求时使用
  - 通过 apt 安装：`sudo apt install python3-venv`
- 注意：如果同时安装了 conda 和 virtualenv，请确认激活了正确的环境（从终端里看，两者可能没有明显区别）

### 容器

- Docker：TODO

## 快速修复

### 更新 CMake

在 Ubuntu 上安装最新版 CMake（由 Claude 生成）：

```sh
sudo apt remove --purge cmake
sudo apt purge cmake
sudo apt autoremove

wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null

sudo add-apt-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
sudo apt update
sudo apt install cmake
```

### 编译器

安装最新版标准库：`sudo apt install libstdc++-12-dev`

### Git 克隆

使用 SSH URL 的一行命令：`sed -i 's/https:\/\/github.com\//git@github.com:/' .gitmodules`

递归处理脚本：

```sh
# Function to update .gitmodules and sync
update_modules() {
    sed -i 's/https:\/\/github.com\//git@github.com:/' .gitmodules
    git submodule sync
    git submodule update --init
}

# Function to process each level
process_level() {
    # 1. Update current .gitmodules
    update_modules

    # 2. For each currently cloned submodule
    git submodule foreach '
        # 3. Update their .gitmodules if exists
        if [ -f ".gitmodules" ]; then
            sed -i "s/https:\/\/github.com\//git@github.com:/" .gitmodules
            # 4. Update their immediate submodules
            git submodule sync
            git submodule update --init
        fi
    '
}

# 5. Repeat multiple times to ensure all nested levels are processed
for i in {1..5}; do
    echo "Processing level $i"
    process_level
done
```

### Hugging Face

HF 镜像：`export HF_ENDPOINT=https://hf-mirror.com`

ModelScope：[modelscope.cn](https://www.modelscope.cn/home)
