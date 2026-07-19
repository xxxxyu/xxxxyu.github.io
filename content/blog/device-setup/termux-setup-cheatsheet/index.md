+++
title = "Cheatsheet for Setting up Termux on Android Smartphones"
date = "2025-01-09"
updated = "2026-07-11"
description = "A practical checklist for setting up Termux as an Android development environment."
path = "blog/termux-setup-cheatsheet"
weight = 30

[extra]
series = "blog/device-setup/_index.md"

[taxonomies]
tags = ["Development", "Android", "Edge Device"]
+++

[Termux](https://termux.dev/en/) provides a Linux-like environment on Android and is much more capable than the ADB shell. As part of the [Smartphone Setup Cheatsheet](/blog/smartphone-setup-cheatsheet/), this article explains how to set it up.

The setup reflects my development needs, which focus on AI computing tasks. It may not suit you if you are primarily interested in app development and GUI debugging.

Related reference: [sanwebinfo/my-termux-setup](https://github.com/sanwebinfo/my-termux-setup)

## Installation

- Install from [F-Droid](https://f-droid.org/) for convenient plugin support
  - Termux: Style
- Install an APK if preferred; downloads are available from [GitHub Releases](https://github.com/termux/termux-app/releases)

## System Setup

System-level setup for Termux.

### Update packages

- Select mirrors automatically: `termux-change-repo`
- Update and upgrade: `pkg update && pkg upgrade`

### Setup SSH

- Install OpenSSH: `pkg install openssh`
- Check Termux username: `whoami`
- Set login password: `passwd`
- Get IP address: `ifconfig`
- Launch SSH server: `sshd`
- Connect from remote: `ssh -p 8022 <username>@<ip>`
  - The default SSH port of Termux is 8022

### Install packages

Common packages for system setup:

- Tsu (su wrapper for Termux): `pkg install tsu`
- Git: `pkg install git`
- Wget: `pkg install wget`
- cURL: `pkg install curl`
- CMake: `pkg install cmake`
- tmux: `pkg install tmux`
- htop: `pkg install htop`

## Environment Setup

Set up development environments.

### Code

Termux does not support the Visual Studio Code remote server, but the open-source [code-server](https://github.com/coder/code-server) is a useful alternative.

Install code-server through the TUR repository:

```sh
pkg install tur-repo
pkg install code-server
```

Configure the server via `~/.config/code-server/config.yaml`. For example:

```yaml
bind-addr: 0.0.0.0:8080
auth: password
```

Launch the server:

```sh
code-server
```

You can now access the code UI at `http://<ip>:8080` with the configured password.

Note that the config path will be slightly different (under `~/.tsu`) if you run in `tsu` mode.

### Python

Termux does not support Miniconda.

Install Python: `pkg install python`
