+++
title = "Cheatsheet for Setting up Termux on Android Smartphones"
date = "2025-01-09"
updated = "2025-01-09"
description = "Quickly setting up Termux on Android smartphones for development."
template = "blog-page.html"

[taxonomies]
tags = ["Development", "Android", "Edge Device"]
+++

[Termux](https://termux.dev/en/) is a useful software for emulating Linux environment in Android (much better than the ADB shell). As part of the [Smartphone Setup Cheatsheet](/blog/smartphone-setup-cheatsheet/), this article elaborates on how to setup Termux.

Also note that this is mostly based on my personal development requirements, which are focused on AI-related computing tasks. It might not perfectly suits you if you're more interested in App development and GUI debugging.

A useful repo: [sanwebinfo/my-termux-setup](https://github.com/sanwebinfo/my-termux-setup)

## Installation

- Install from [F-Droid](https://f-droid.org/) to get convenient plugin support
  - Termux: Style
- Install from APK if you prefer. Download from [GitHub Release](https://github.com/termux/termux-app/releases)

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

Common packages during system setup:

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

Termux doesn't support VSC (Visual Studio Code) remote server, but you can set up an open-source [code-server](https://github.com/coder/code-server) as an alternative.

Install code-server via Tur repo:

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

Now you will be able to access the code UI via <http://ip:8080> with the configured password.

Note that the config path will be slightly different (under `~/.tsu`) if you run in `tsu` mode.

### Python

Termux doesn't support miniconda.

Install Python: `pkg install python`
