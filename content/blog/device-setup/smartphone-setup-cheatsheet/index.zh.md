+++
title = "Android 智能手机配置速查表"
date = "2025-01-09"
updated = "2026-07-11"
description = "面向开发工作的 Android 智能手机配置实用检查清单。"
path = "zh/blog/smartphone-setup-cheatsheet"
weight = 20

[extra]
series = "blog/device-setup/_index.md"
ai_translation_source = "en"

[taxonomies]
tags = ["Development", "Android", "Edge Device"]
+++

这份速查表源于我自己的开发需求，主要面向 AI 计算任务，以及在类 Linux 环境中运行 DNN。如果你更关心应用开发和 GUI 调试，它可能并不适用。

## 首次开机

### 注册

注册新设备以激活保修（如果你不打算获取系统 root 权限）。

## 获取系统 Root 权限

如需以超级用户身份深度控制系统，必须取得 root 权限。

[Magisk 中文网教程](https://magiskcn.com)

本节以一加设备为例，因为在国内，一加对 root 相对友好。

所需工具与文件：

- ADB（Android Debug Bridge）：[platform-tools](https://developer.android.com/tools/releases/platform-tools)
- Android OTA payload 提取工具：[payload-dumper-go](https://github.com/ssut/payload-dumper-go)
- 系统 ROM：[MagiskCN](https://magiskcn.com/roms.html)
- Magisk APK：[GitHub Release](https://github.com/topjohnwu/Magisk/releases)

### 解锁 Bootloader

[Magisk 中文网教程](https://magiskcn.com/oneplus-unlock-qualcomm#google_vignette)

1. 进入开发者模式
2. 开启 OEM 解锁（具体方式取决于厂商）
3. 开启 USB 调试，将手机连接至主机
4. 进入 bootloader：`adb reboot bootloader`
5. 解锁 bootloader：`fastboot flashing unlock`
6. 在手机上选择“解锁 bootloader”并确认

### 使用 Magisk 获取 Root 权限

[Magisk 中文网教程](https://magiskcn.com/oneplus-init-boot)

1. 在主机上下载匹配的 ROM 包并解压
2. 使用 payload 提取工具从 ROM 中提取 `init_boot.img`
   - 用法：`payload-dumper-go -p init_boot <extracted>/payload.bin`
3. 开启 USB 调试，将手机连接至主机
4. 将 `init_boot.img` 传到手机的“Download”文件夹
5. 在手机上安装 Magisk 应用
6. 在 Magisk 中选择“安装”，修补 `init_boot.img`，得到 `magisk_patched-xxx.img`
7. 将 `magisk_patched-xxx.img` 传回主机
8. 进入 bootloader：`adb reboot bootloader`
9. 刷入补丁：`fastboot flash init_boot magisk_patched-xxx.img`
    - 成功时应看到“Okay”
10. 重启系统：`fastboot reboot`

重启后打开 Magisk，查看其中显示的版本信息，确认 root 权限已获取成功。

## 系统设置

### 开发者选项

- 开启开发者选项
- 开启 USB 调试
- 开启无线调试（如有需要）

### 系统更新

建议关闭所有可能影响系统性能与版本稳定性的设置，具体包括：

- 关闭不必要的后台服务
- 关闭系统自动更新与自动下载
- 移除自动安装的应用

### 电池管理

- 在系统全局电池设置中开启“性能模式”
- 在各应用的电池设置中开启“允许后台运行”
- 为 Termux 及类似进程选择“获取唤醒锁”

## 应用

安装方式：

- 从应用商店安装（Google Play、F-Droid 等）
- 安装已下载的 APK（Android Application Package）
- 通过 ADB 从主机安装（需要 APK）

### Clash for Android（CFA）

首先配置代理。

TODO：补充细节

### F-Droid

[F-Droid](https://f-droid.org/) 是一个可安装的 Android FOSS（自由及开放源代码软件）应用目录，也可作为 Google Play 的替代方案。

### Termux

推荐通过 F-Droid 安装。

另请参阅 [Termux 配置速查表](/zh/blog/termux-setup-cheatsheet/)。

### Microsoft Edge

可用它替代功能可能受限的系统浏览器。

我推荐 Edge，是因为国内各大应用商店通常都能下载到它。
