+++
title = "Android 手机上 Termux 配置速查表"
date = "2025-01-09"
updated = "2026-07-11"
description = "将 Termux 配置为 Android 开发环境的实用检查清单。"
path = "zh/blog/termux-setup-cheatsheet"
weight = 30

[extra]
series = "blog/device-setup/_index.md"
ai_translation_source = "en"

[taxonomies]
tags = ["Development", "Android", "Edge Device"]
+++

[Termux](https://termux.dev/en/) 在 Android 上提供了一个类 Linux 环境，功能比 ADB shell 完整得多。作为[智能手机配置速查表](/zh/blog/smartphone-setup-cheatsheet/)的延伸，本文说明如何配置 Termux。

这套配置服务于我以 AI 计算任务为主的开发需求。如果你主要关注应用开发和 GUI 调试，它可能并不完全适合。

相关参考：[sanwebinfo/my-termux-setup](https://github.com/sanwebinfo/my-termux-setup)

## 安装

- 推荐从 [F-Droid](https://f-droid.org/) 安装，便于使用插件
  - Termux: Style
- 也可以直接安装 APK；下载见 [GitHub Releases](https://github.com/termux/termux-app/releases)

## 系统配置

以下是在 Termux 内进行的系统级配置。

### 更新软件包

- 自动选择镜像：`termux-change-repo`
- 更新并升级：`pkg update && pkg upgrade`

### 配置 SSH

- 安装 OpenSSH：`pkg install openssh`
- 查看 Termux 用户名：`whoami`
- 设置登录密码：`passwd`
- 获取 IP 地址：`ifconfig`
- 启动 SSH 服务：`sshd`
- 从远端连接：`ssh -p 8022 <username>@<ip>`
  - Termux 的默认 SSH 端口是 8022

### 安装常用软件包

系统配置常用的软件包：

- Tsu（Termux 的 su 封装）：`pkg install tsu`
- Git：`pkg install git`
- Wget：`pkg install wget`
- cURL：`pkg install curl`
- CMake：`pkg install cmake`
- tmux：`pkg install tmux`
- htop：`pkg install htop`

## 开发环境配置

配置开发所需的环境。

### Code

Termux 不支持 Visual Studio Code remote server，但开源的 [code-server](https://github.com/coder/code-server) 是一个实用的替代方案。

通过 TUR 仓库安装 code-server：

```sh
pkg install tur-repo
pkg install code-server
```

在 `~/.config/code-server/config.yaml` 中配置服务。例如：

```yaml
bind-addr: 0.0.0.0:8080
auth: password
```

启动服务：

```sh
code-server
```

现在可以通过 `http://<ip>:8080` 访问代码编辑界面，并使用配置的密码登录。

如果在 `tsu` 模式下运行，配置文件路径会略有不同（位于 `~/.tsu` 下）。

### Python

Termux 不支持 Miniconda。

安装 Python：`pkg install python`
