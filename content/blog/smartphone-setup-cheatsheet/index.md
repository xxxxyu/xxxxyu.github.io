+++
title = "Cheatsheet for Setting up Android Smartphones"
date = "2025-01-09"
updated = "2026-07-11"
description = "A practical checklist for setting up Android smartphones for development."
template = "blog-page.html"

[taxonomies]
tags = ["Development", "Android", "Edge Device"]
+++

This cheatsheet reflects my development needs, which focus on AI computing tasks and running DNNs in a Linux-like environment. It may not suit you if you are primarily interested in app development and GUI debugging.

## First Boot

### Register

Register your new device to activate the warranty (if you are not going to root the system).

## Root the System

Root access is required for advanced system control as the superuser.

[Magisk 中文网教程](https://magiskcn.com)

This section uses a OnePlus device as an example because the brand is relatively root-friendly in China.

Requirements:

- ADB (Android Debug Bridge): [platform-tools](https://developer.android.com/tools/releases/platform-tools)
- Android OTA payload dumper: [payload-dumper-go](https://github.com/ssut/payload-dumper-go)
- System ROM: [MagiskCN](https://magiskcn.com/roms.html)
- Magisk APK: [GitHub Release](https://github.com/topjohnwu/Magisk/releases)

### Unlock Bootloader

[Magisk 中文网教程](https://magiskcn.com/oneplus-unlock-qualcomm#google_vignette)

1. Enter developer mode
2. OEM unlock (depends on vendors)
3. Connect to the host PC with USB debugging
4. Enter bootloader: `adb reboot bootloader`
5. Unlock the bootloader: `fastboot flashing unlock`
6. On the smartphone, select "unlock the bootloader" and confirm.

### Root with Magisk

[Magisk 中文网教程](https://magiskcn.com/oneplus-init-boot)

1. On the host PC, download the matching ROM package and extract
2. Extract `init_boot.img` from the ROM with the payload dumper
   - Usage: `payload-dumper-go -p init_boot <extracted>/payload.bin`
3. Connect to the smartphone with USB debugging
4. Transfer `init_boot.img` to the smartphone's "Download" folder
5. Install the Magisk app on the smartphone
6. In the Magisk app, select "Install" and patch `init_boot.img` to obtain `magisk_patched-xxx.img`
7. Transfer `magisk_patched-xxx.img` back to the host PC
8. Enter bootloader: `adb reboot bootloader`
9. Flash the patch: `fastboot flash init_boot magisk_patched-xxx.img`
    - You should see "Okay" if it succeeds.
10. Reboot the system: `fastboot reboot`

After rebooting, open the Magisk app and check the displayed version to confirm successful root access.

## System Settings

### Developer Options

- Turn on developer options
- Turn on USB debugging
- Turn on wireless debugging (if you prefer)

### System Update

It is recommended to turn off all settings that might influence system performance and version stability. Specifically:

- Turn off unnecessary background services
- Turn off system auto-update and auto-download
- Remove auto-installed apps

### Battery Management

- Turn on "performance mode" in system-wide battery settings
- Turn on "allow background" in per-app battery settings
- Select "acquire wakelock" for Termux and similar processes

## Apps

How to install:

- Install from app stores (Google Play, F-Droid, etc.)
- Install a downloaded APK (Android Application Package)
- Install from host PC by ADB (APK required)

### Clash for Android (CFA)

First, set up the proxy.

TODO: details

### F-Droid

[F-Droid](https://f-droid.org/) is an installable catalog of FOSS (Free and Open Source Software) Android applications and an alternative to Google Play.

### Termux

Installation through F-Droid is recommended.

Check out the [Termux Setup Cheatsheet](/blog/termux-setup-cheatsheet/).

### Microsoft Edge

Use it as an alternative to the system browser, which may have limited functionality.

I recommend Edge because it is widely available in Chinese app stores.
