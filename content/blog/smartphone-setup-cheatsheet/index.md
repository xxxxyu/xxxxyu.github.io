+++
title = "Cheatsheet for Setting up Android Smartphones"
date = "2025-01-09"
updated = "2025-08-04"
description = "Quickly setting up Android smartphones for development."
template = "blog-page.html"

[taxonomies]
tags = ["Development", "Android", "Edge Device"]
+++

This is mostly based on my personal development requirements, which are focused on AI-related computing tasks. Specifically, I run DNNs in a Linux-like environment. So it might not perfectly suits you if you're more interested in App development and GUI debugging.

## First Boot

### Register

Register your new device to activate the warranty (if you are not going to root the system).

## Root the System

Root is required to obtain advanced system control (as super user).

Magisk 中文网教程：<https://magiskcn.com>

This section continues with an OnePlus example, as it is the most root-friendly Chinese smartphone brand.

Requirements:

- ADB (Android Debug Bridge): [platform-tools](https://developer.android.com/tools/releases/platform-tools)
- Android OTA payload dumper: [payload-dumper-go](https://github.com/ssut/payload-dumper-go)
- System ROM: [MagiskCN](https://magiskcn.com/roms.html)
- Magisk APK: [GitHub Release](https://github.com/topjohnwu/Magisk/releases)

### Unlock Bootloader

Magisk 中文网教程：<https://magiskcn.com/oneplus-unlock-qualcomm#google_vignette>

1. Enter developer mode
2. OEM unlock (depends on vendors)
3. Connect to the host PC with USB debugging
4. Enter bootloader: `adb reboot bootloader`
5. Unlock bootloader: `bootloader flashing unlock`
6. On the smartphone, select "unlock the bootloader" and confirm.

### Root with Magisk

Magisk 中文网教程：<https://magiskcn.com/oneplus-init-boot>

1. On the host PC, download the matching ROM package and extract
2. Extract init_boot.img from the ROM with the payload dumper
   - Usage: `payload-dumper-go -p init_boot <extracted>/payload.bin`
3. Connect to the smartphone with USB debugging
4. Transfer init_boot.img to the smartphone's "Download" folder
5. Install Magisk app to the smartphone
6. In Magisk app, "Install" init_boot.img to obtain "magisk_patched-xxx.img"
7. Transfer "magisk_patched-xxx.img" back to the host PC
8. Enter bootloader: `adb reboot bootloader`
9. Flash the patch: `fastboot flash init_boot magisk_patched-xxx.img`
    - You should see "Okay" if it succeeds.
10. Reboot the system: `fastboot reboot`

After rebooting, enter the Magisk app and check the "Magisk" version. You should be able to see the correct version displayed after a successful rooting.

## System Setting

### Developer Option

- Turn on developer option
- Turn on USB debugging
- Turn on wireless debugging (if you prefer)

### System Update

It is recommended to turn off all settings that might influence system performance and version stability. Specifically:

- Turn off unecessary (background) services
- Turn off system auto-update and auto-download
- Remove auto-installed apps

### Battery Management

- Turn on "performance mode" in system-wide battery settings
- Turn on "allow background" in app-wise battery settings
- Select "acquire wakelock" for Termux processes and alike

## Apps

How to install:

- Install from app stores (Google Play, F-Droid, etc.)
- Install from web-downloaded APK (Android Application Package)
- Install from host PC by ADB (APK required)

### Clash for Android (CFA)

First and foremost, setup the proxy.

TODO: details

### F-Droid

F-Droid is an installable catalogue of FOSS (Free and Open Source Software) applications for the Android platform. Use it as an alternative to Google Play.

F-Droid site: <https://f-droid.org/>

### Termux

Installation by F-Droid is recommended.

Check out the [Termux Setup Cheatsheet](/blog/termux-setup-cheatsheet/).

### Microsoft Edge

Use it as an alternative to the system default browser, which often has limited functionalities.

I recommend Edge as it is the most available one (in Chinese app stores).
