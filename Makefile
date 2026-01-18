# Rusty Eyes 3 - Build Automation

.PHONY: all setup-apt setup-dnf build run check-health

all: build

# Debian/Ubuntu Setup
setup-apt:
	@echo "Installing dependencies for Debian/Ubuntu..."
	sudo apt-get update
	sudo apt-get install -y build-essential libssl-dev pkg-config python3-dev python3-pip protobuf-compiler
	# GStreamer dependencies
	sudo apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
		libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base \
		gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
		gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools \
		gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl \
		gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio \
		libgstrtspserver-1.0-dev
	# X11 dependencies (for overlay)
	sudo apt-get install -y libx11-dev libxfixes-dev

# Fedora/RHEL Setup
setup-dnf:
	@echo "Installing dependencies for Fedora/RHEL..."
	sudo dnf groupinstall -y "C Development Tools and Libraries"
	sudo dnf install -y openssl-devel pkgconfig python3-devel protobuf-compiler
	# GStreamer dependencies
	sudo dnf install -y gstreamer1-devel gstreamer1-plugins-base-devel \
		gstreamer1-plugins-good gstreamer1-plugins-bad-free \
		gstreamer1-plugins-ugly-free gstreamer1-libav gstreamer1-tools \
		gstreamer1-plugins-base-tools gstreamer1-rtsp-server-devel
	# X11 dependencies
	sudo dnf install -y libX11-devel libXfixes-devel

# Build Rust Project
build:
	cargo build --release --no-default-features

# Run Application
run:
	cargo run --release --no-default-features

# Remote Server Utils
setup-server:
	cd remote_server && pip3 install -r requirements.txt
	cd remote_server && ./deploy.sh
