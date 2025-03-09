OverlayFS (Overlay Filesystem) is a union filesystem used in Linux to merge multiple directories so that they appear as a single directory. 
Used in: containers (Docker), LiveCDs, snapshot systems

## Basic Concepts 
OverlayFS overlays two directories (called "layers") together:
- Upper Layer: The writable layer where all new files or modifications are stored.
- Lower Layer: The read-only layer that provides the base data and cannot be modified.

When you modify a file:
- If the file originally exists in the Lower Layer, it will be copied to the Upper Layer (Copy-on-Write, CoW) and modified there.
- If the file already exists in the Upper Layer, it will be modified directly.

## Key Features
- Lightweight: does not require copying the entire filesystem.
- Copy-on-Write (CoW): Avoids modifying original data, preserving the integrity of the lower layer.
- Multi-layer support: Technologies like Docker use multiple layers to build images, improving storage and management efficiency.
- High performance: has a simpler structure and lower overhead, making it ideal for container environments.


## Example
Assume you have two directories:
- /upperdir: The read-only lower layer.
- /lowerdir: The writable upper layer.
- /merged: The final merged mount point.
To create an OverlayFS mount:
`sudo mount -t overlay -o lowerdir=/lowerdir,upperdir=/upperdir,workdir=/workdir overlay /merged`
Note that workdir is a required working directory for OverlayFS. It must be an empty directory and on the same filesystem as upperdir.

eg, mount a data-vol to system-vol
`sudo mount -t overlay -o lowerdir=system-vol,upperdir=data-vol/data,workdir=data-vol/work overlay system-vol`

umount
`sudo umount /merged`
`sudo umount -l system-vol  # Lazy unmount (will unmount when not in use)`
`sudo umount -f system-vol  # Force unmount (use with caution)`